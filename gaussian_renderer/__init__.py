import torch
import math

import numpy as np
from complex_gaussian_tracer import ComplexGaussianTracerSettings, ComplexGaussianTracer
from scene.gaussian_model import GaussianModel
from utils.fle_utils import eval_fle


# ---- covariance utilities ----

def _rotate_covariance(cov6, R):
    """Rotate (N,6) upper-triangle covariance by R: Cov_new = R @ Cov @ R^T"""
    N = cov6.shape[0]
    full = _rebuild_symmetric(cov6)
    rotated = R @ full @ R.T
    out = torch.zeros_like(cov6)
    out[:, 0] = rotated[:, 0, 0]
    out[:, 1] = rotated[:, 0, 1]
    out[:, 2] = rotated[:, 0, 2]
    out[:, 3] = rotated[:, 1, 1]
    out[:, 4] = rotated[:, 1, 2]
    out[:, 5] = rotated[:, 2, 2]
    return out


def _rebuild_symmetric(cov6):
    """Rebuild (N,3,3) symmetric matrices from (N,6) upper triangle."""
    N = cov6.shape[0]
    full = torch.zeros(N, 3, 3, device=cov6.device, dtype=cov6.dtype)
    full[:, 0, 0] = cov6[:, 0]
    full[:, 0, 1] = cov6[:, 1]
    full[:, 0, 2] = cov6[:, 2]
    full[:, 1, 0] = cov6[:, 1]
    full[:, 1, 1] = cov6[:, 3]
    full[:, 1, 2] = cov6[:, 4]
    full[:, 2, 0] = cov6[:, 2]
    full[:, 2, 1] = cov6[:, 4]
    full[:, 2, 2] = cov6[:, 5]
    return full


# ---- ray generation ----

def calculate_gaussian_radii(full_cov_matrices, scale=3.0):
    """Compute bounding radii from covariance eigenvalues."""
    eigenvalues, _ = torch.linalg.eigh(full_cov_matrices)
    max_eigenvalues = torch.max(eigenvalues, dim=-1)[0]
    radii = scale * torch.sqrt(max_eigenvalues)
    return radii


def create_sphere_rays(n_azimuth, n_elevation, radius=0.5):
    """Generate ray directions on a sphere grid (azimuth x elevation)."""
    azimuth = torch.linspace(1, 360, n_azimuth) / 180 * np.pi
    elevation = torch.linspace(1, 90, n_elevation) / 180 * np.pi

    azimuth = torch.tile(azimuth, (n_elevation,))
    elevation = torch.repeat_interleave(elevation, n_azimuth)

    x = radius * torch.cos(elevation) * torch.cos(azimuth)
    y = radius * torch.cos(elevation) * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    r_d = torch.stack([x, y, z], dim=0)
    return r_d


def select_coarse_ray_indices(n_azimuth, n_elevation, step):
    """Downsample ray grid by selecting center indices at given step size."""
    def calculate_centers(n, step):
        centers = []
        for start in range(0, n, step):
            end = min(start + step, n)
            center = (start + end - 1) // 2
            centers.append(center)
        return centers

    azimuth_centers = calculate_centers(n_azimuth, step)
    elevation_centers = calculate_centers(n_elevation, step)

    representatives = []
    for elevation_idx in elevation_centers:
        for azimuth_idx in azimuth_centers:
            index = elevation_idx * n_azimuth + azimuth_idx
            representatives.append(index)

    return representatives


# ---- RFID rendering ----

def render_rfid(viewpoint,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor
           ):
    scaling_modifier = 1.0
    radii_scale      = 3.0

    radius_rx = pipe.radius_rx

    # extract Gaussian properties
    means_3d    = pc.get_xyz
    fle_coeffs  = pc.get_features
    attenuation = pc.get_attenuation

    cov3d_precomp, actual_cov3d = pc.get_covariance(scaling_modifier)

    # TX/RX positions
    tvec_sphere_center = viewpoint.T_tx.to(means_3d.device, dtype=means_3d.dtype)
    tvec_rx            = viewpoint.T_rx.to(means_3d.device, dtype=means_3d.dtype)

    # generate sphere ray directions centered at TX
    n_azimuth = int(viewpoint.width)
    n_elevation = int(viewpoint.height)
    r_d_fine_ori = create_sphere_rays(n_azimuth=n_azimuth,
                                              n_elevation=n_elevation,
                                              radius=radius_rx).to(means_3d.device, dtype=means_3d.dtype)

    r_d_fine_t = r_d_fine_ori + tvec_sphere_center[:, None]
    r_d_w_fine = r_d_fine_t.permute(1, 0)

    # evaluate FLE (Fourier-Legendre Expansion) for directional radiance
    fle_view = fle_coeffs.transpose(1, 2).view(-1, pc.num_channels, (pc.max_fle_degree + 1) ** 2)

    dir_pp = (means_3d - tvec_sphere_center.repeat(means_3d.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    fle_re, fle_im = eval_fle(pc.active_fle_degree, fle_view, dir_pp_normalized)

    # RX modulation: path loss + scattering (only for multi-RX datasets like BLE)
    rx_modulation_enabled = getattr(pipe, 'rx_modulation', False)
    if rx_modulation_enabled:
        dir_to_rx = (tvec_rx.repeat(means_3d.shape[0], 1) - means_3d)
        dist_to_rx = dir_to_rx.norm(dim=1).clamp(min=1e-6)
        dir_to_rx_normalized = dir_to_rx / dist_to_rx.unsqueeze(1)

        rx_path_loss = 1.0 / dist_to_rx
        rx_path_loss = rx_path_loss / rx_path_loss.mean()

        scatter_cos = (dir_pp_normalized * dir_to_rx_normalized).sum(dim=1)
        scatter_weight = (1.0 + scatter_cos) / 2.0

        rx_modulation = rx_path_loss * scatter_weight
        fle_re = fle_re * rx_modulation
        fle_im = fle_im * rx_modulation

    # coarse ray subset (required by CUDA interface)
    coarse_step = max(1, min(n_azimuth, n_elevation) // 4)
    idx_list = select_coarse_ray_indices(n_azimuth=n_azimuth, n_elevation=n_elevation, step=coarse_step)
    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    r_d_w_coarse = r_d_w_fine[idx_tensor]

    # placeholder for CUDA interface
    tvec_cond_embd = torch.zeros(63, device=means_3d.device, dtype=means_3d.dtype)

    radii = calculate_gaussian_radii(actual_cov3d, scale=radii_scale)

    # build rasterizer settings and run complex gaussian tracer
    raster_settings_t = ComplexGaussianTracerSettings(height=int(viewpoint.height),
                                                      width=int(viewpoint.width),
                                                      fle_degree_active=pc.active_fle_degree,
                                                      spectrum_3d_coarse=r_d_w_coarse,
                                                      spectrum_3d_fine=r_d_w_fine,
                                                      sphere_center=tvec_sphere_center,
                                                      sphere_radius=radius_rx,
                                                      cond_embd=tvec_cond_embd,
                                                      bg=bg_color,
                                                      debug=pipe.debug,
                                                      gaus_radii=radii
                                                      )

    rasterizer = ComplexGaussianTracer(raster_settings=raster_settings_t)

    # stack Re + Im complex signal for CUDA
    stacked_signal = torch.stack((fle_re, fle_im), dim=1)

    rendered_image_complex = rasterizer(means_3d=means_3d,
                                cov3d_precomp=cov3d_precomp,
                                signal_precomp=stacked_signal,
                                attenuation=attenuation,
                                )

    # compute amplitude from complex output
    real_part      = rendered_image_complex[0, :, :]
    imaginary_part = rendered_image_complex[1, :, :]
    rendered_image = torch.sqrt(real_part**2 + imaginary_part**2 + 1e-8)

    return {"render":            rendered_image,
            "visibility_filter": radii > 0.0,
            "radii":             radii,
            }
