import torch
import math

import numpy as np
import torch.nn.functional as F

from complex_gaussian_tracer import ComplexGaussianTracerSettings, ComplexGaussianTracer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.pos_encoder import Embedder

import time
from scipy.linalg import eigh

import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, size):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.norm(sublayer(x))


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForward, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


class SingleMLPWithAddNorm(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(SingleMLPWithAddNorm, self).__init__()
        self.feed_forward1 = FeedForward(input_size, hidden_size)
        self.feed_forward2 = FeedForward(hidden_size, hidden_size)
        self.feed_forward3 = FeedForward(hidden_size, input_size)
        
        self.add_norm2 = AddNorm(hidden_size)
        
    def forward(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        
        # Flatten the input tensor
        original_shape = x.shape
        x = x.view(1, -1)
        
        x = self.feed_forward1(x)

        x = self.add_norm2(x, self.feed_forward2)
        x = self.feed_forward3(x)
        
        x = x.view(original_shape)
        
        return x


def calculate_gaussian_radii(full_cov_matrices, scale=3.0):

    # Compute eigenvalues for each matrix
    eigenvalues, _ = torch.linalg.eigh(full_cov_matrices)

    # Find the maximum eigenvalues along the last dimension (axis=-1)
    max_eigenvalues = torch.max(eigenvalues, dim=-1)[0]

    # Compute the radii as 3 times the square root of these eigenvalues
    radii = scale * torch.sqrt(max_eigenvalues)

    return radii


def create_ray_direction_fine_v2(n_azimuth=360, n_elevation=90, radius=0.5):

    # Compute azimuth and elevation angles
    azimuth = torch.linspace(1, 360, n_azimuth) / 180 * np.pi
    elevation = torch.linspace(1, 90, n_elevation) / 180 * np.pi

    # Repeat azimuth for each elevation
    azimuth = torch.tile(azimuth, (n_elevation,))

    # Repeat each elevation value for each azimuth
    elevation = torch.repeat_interleave(elevation, n_azimuth)

    x = radius * torch.cos(elevation) * torch.cos(azimuth)
    y = radius * torch.cos(elevation) * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    r_d = torch.stack([x, y, z], dim=0)

    return r_d


def calculate_midpoints(total_degrees, block_size):

    centers = []
    for start in range(1, total_degrees + 1, block_size):
        end = min(start + block_size - 1, total_degrees)
        center = (start + end) / 2.0
        centers.append(center)

    return torch.tensor(centers)


def select_representative_directions_idx(n_azimuth=360, n_elevation=90, step=16):

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


def render(viewpoint, 
           pc : GaussianModel, 
           pos_enc: Embedder,
           pipe, 
           bg_color : torch.Tensor
           ):
    scaling_modifier = 1.0
    radii_scale      = 3.0

    radius_rx = pipe.radius_rx

    means_3d    = pc.get_xyz   
    shs_coeffs  = pc.get_features
    attenuation = pc.get_attenuation

    cov3d_precomp, actual_cov3d = pc.get_covariance(scaling_modifier)

    # not unsed in CUDA code
    radii = calculate_gaussian_radii(actual_cov3d, scale=radii_scale)

    rotation_rx = viewpoint.R.to(means_3d.device, dtype=means_3d.dtype)

    tvec_rx    = viewpoint.T_tx
    tvec_tx    = viewpoint.T_rx

    tvec_rx = tvec_rx.to(means_3d.device, dtype=means_3d.dtype)
    tvec_tx = tvec_tx.to(means_3d.device, dtype=means_3d.dtype)

    r_d_fine_ori = create_ray_direction_fine_v2(n_azimuth=360,
                                              n_elevation=90,
                                              radius=radius_rx).to(means_3d.device, dtype=means_3d.dtype)
    
    # it seems does not work well, has poor performance
    r_d_w_fine_rotation = rotation_rx @ r_d_fine_ori

    r_d_fine_t = r_d_fine_ori + tvec_rx[:, None]

    r_d_w_fine = r_d_fine_t.permute(1, 0)

    shs_view = shs_coeffs.transpose(1, 2).view(-1, pc.num_channels, (pc.max_sh_degree + 1) ** 2)

    dir_pp = (means_3d - tvec_rx.repeat(means_3d.shape[0], 1))         
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # not unsed in CUDA code
    idx_list = select_representative_directions_idx(n_azimuth=360, n_elevation=90, step=16)
    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    r_d_w_coarse = r_d_w_fine[idx_tensor]

    tvec_tx_re = torch.reshape(tvec_tx, [-1, 3]).float()                  
    tvec_tx_embd = pos_enc(tvec_tx_re)                                  
    tvec_tx_embd = tvec_tx_embd.squeeze(0)

    raster_settings_t = ComplexGaussianTracerSettings(height=int(viewpoint.height),
                                                      width=int(viewpoint.width),
                                                      sh_degree_active=pc.active_sh_degree,
                                                      spectrum_3d_coarse=r_d_w_coarse,  
                                                      spectrum_3d_fine=r_d_w_fine,      
                                                      rx_pos=tvec_rx,                   
                                                      radius_rx=radius_rx,              
                                                      tx_pos=tvec_tx_embd,
                                                      bg=bg_color,
                                                      debug=pipe.debug,
                                                      gaus_radii=radii
                                                      )

    rasterizer = ComplexGaussianTracer(raster_settings=raster_settings_t)

    singal_amp, singal_pha = colors_precomp[:, 0], colors_precomp[:, 1]
    singal_amp     = abs(F.leaky_relu(singal_amp))
    singal_pha     = torch.sigmoid(singal_pha) * np.pi * 2
    stacked_signal = torch.stack((singal_amp, singal_pha), dim=1)

    rendered_image_complex = rasterizer(means_3d=means_3d,                
                                cov3d_precomp=cov3d_precomp, 
                                signal_precomp=stacked_signal,
                                attenuation=attenuation,
                                )



    real_part      = rendered_image_complex[0, :, :]                 
    imaginary_part = rendered_image_complex[1, :, :]
    rendered_image = torch.sqrt(real_part**2 + imaginary_part**2)   

    return {"render":            rendered_image,    
            "visibility_filter": radii > 0.0,       
            "radii":             radii,        
            }



