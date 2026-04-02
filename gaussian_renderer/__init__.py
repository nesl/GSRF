import torch
import numpy as np
from complex_gaussian_tracer import ComplexGaussianTracerSettings, ComplexGaussianTracer
from scene.gaussian_model import GaussianModel
from utils.fle_utils import eval_fle


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




# ---- RFID rendering ----

def render_rfid(viewpoint,
           pc : GaussianModel,
           pipe
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

    radii = calculate_gaussian_radii(actual_cov3d, scale=radii_scale)

    # build rasterizer settings and run complex gaussian tracer
    raster_settings_t = ComplexGaussianTracerSettings(height=int(viewpoint.height),
                                                      width=int(viewpoint.width),
                                                      fle_degree_active=pc.active_fle_degree,
                                                      spectrum_3d_fine=r_d_w_fine,
                                                      sphere_center=tvec_sphere_center,
                                                      sphere_radius=radius_rx,
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
