
import torch
import numpy as np

from complex_gaussian_tracer_csi import ComplexGaussianTracerSettings, ComplexGaussianTracer
from utils.fle_utils import eval_fle


# ---- ray generation ----

def create_sphere_rays(n_azimuth=36, n_elevation=9, radius=0.5):
    """Generate ray directions on a sphere grid (full elevation range -89 to 89)."""
    azimuth = torch.linspace(1, 360, n_azimuth) / 180 * np.pi
    elevation = torch.linspace(-89, 89, n_elevation) / 180 * np.pi

    azimuth = torch.tile(azimuth, (n_elevation,))
    elevation = torch.repeat_interleave(elevation, n_azimuth)

    x = radius * torch.cos(elevation) * torch.cos(azimuth)
    y = radius * torch.cos(elevation) * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    return torch.stack([x, y, z], dim=0)


from gaussian_renderer import calculate_gaussian_radii
from utils.fle_utils import _associated_legendre_batch, _build_norm_table


def eval_fle_multi_channel(deg, coeffs, dirs):
    """Evaluate FLE for multi-channel CSI (26 subcarriers x 2).

    Unlike eval_fle() which handles 2 channels (single Re/Im pair),
    this handles 52 channels with interleaved real/imaginary coefficients
    per subcarrier using batched einsum.
    """
    n_coeffs = (deg + 1) ** 2

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    alpha = torch.atan2(y, x)
    cos_theta = z

    plm = _associated_legendre_batch(deg, cos_theta)
    norm = _build_norm_table(deg)

    basis_re_list = []
    basis_im_list = []
    for l in range(deg + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            bp = norm[(l, m)] * plm[l][abs_m]
            if m < 0:
                bp = bp * ((-1) ** abs_m)
            cos_ma = torch.cos(m * alpha)
            sin_ma = torch.sin(m * alpha)
            basis_re_list.append(cos_ma * bp)
            basis_im_list.append(sin_ma * bp)

    B_re = torch.stack(basis_re_list, dim=-1)
    B_im = torch.stack(basis_im_list, dim=-1)

    # per-subcarrier signal: interleaved real (even) and imaginary (odd) coefficients
    coeffs_all = coeffs[:, :, :n_coeffs]
    a = coeffs_all[:, 0::2, :]
    b = coeffs_all[:, 1::2, :]

    sig_re = torch.einsum('nci,ni->nc', a, B_re) - torch.einsum('nci,ni->nc', b, B_im)
    sig_im = torch.einsum('nci,ni->nc', a, B_im) + torch.einsum('nci,ni->nc', b, B_re)

    return sig_re, sig_im


# cache for ray directions
_cache = {}


# ---- CSI rendering ----

def render_csi(viewpoint, pc, pipe, n_azimuth=36, n_elevation=9):
    scaling_modifier = 1.0
    radii_scale = 3.0
    radius_rx = pipe.radius_rx

    # extract Gaussian properties
    means_3d = pc.get_xyz
    fle_coeffs = pc.get_features
    attenuation = pc.get_attenuation

    cov3d_precomp, actual_cov3d = pc.get_covariance(scaling_modifier)

    # TX/RX positions
    tvec_sphere_center = viewpoint.T_tx.to(means_3d.device, dtype=means_3d.dtype)
    tvec_rx = viewpoint.T_rx.to(means_3d.device, dtype=means_3d.dtype)

    # cache ray directions and coarse indices per (resolution, radius, device)
    cache_key = (n_azimuth, n_elevation, radius_rx, means_3d.device)
    if cache_key not in _cache:
        _cache[cache_key] = {
            'r_d_fine_ori': create_sphere_rays(
                n_azimuth=n_azimuth, n_elevation=n_elevation, radius=radius_rx
            ).to(means_3d.device, dtype=means_3d.dtype),
        }

    cached = _cache[cache_key]
    r_d_fine_ori = cached['r_d_fine_ori']

    # translate rays to TX position
    r_d_fine_t = r_d_fine_ori + tvec_sphere_center[:, None]
    r_d_w_fine = r_d_fine_t.permute(1, 0)

    # evaluate FLE for per-subcarrier complex signal
    fle_view = fle_coeffs.transpose(1, 2).view(-1, pc.num_channels, (pc.max_fle_degree + 1) ** 2)

    dir_pp = (means_3d - tvec_sphere_center.repeat(means_3d.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp(min=1e-6)

    sig_re, sig_im = eval_fle_multi_channel(pc.active_fle_degree, fle_view, dir_pp_normalized)
    stacked_signal = torch.stack([sig_re, sig_im], dim=2).reshape(-1, 52)

    radii = calculate_gaussian_radii(actual_cov3d, scale=radii_scale)

    # build rasterizer settings and run complex gaussian tracer
    raster_settings = ComplexGaussianTracerSettings(
        height=n_elevation,
        width=n_azimuth,
        fle_degree_active=pc.active_fle_degree,
        spectrum_3d_fine=r_d_w_fine,
        sphere_center=tvec_sphere_center,
        sphere_radius=radius_rx,
        debug=pipe.debug,
        gaus_radii=radii,
    )

    rasterizer = ComplexGaussianTracer(raster_settings=raster_settings)

    rendered = rasterizer(
        means_3d=means_3d,
        cov3d_precomp=cov3d_precomp,
        signal_precomp=stacked_signal,
        attenuation=attenuation,
    )

    # unit ray directions for downstream CSI processing
    ray_unit = r_d_fine_ori / r_d_fine_ori.norm(dim=0, keepdim=True).clamp(min=1e-6)
    ray_unit = ray_unit.permute(1, 0)

    return {
        "render": rendered,
        "ray_dirs": ray_unit,
        "visibility_filter": radii > 0.0,
        "radii": radii,
    }
