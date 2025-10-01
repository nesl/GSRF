#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def tracer_complex_gaussians(means_3d,
                        cov3d_precomp,
                        signal_precomp,
                        attenuation,
                        raster_settings,
                        ):
    
    return _TracerComplexGaussians.apply(means_3d,
                                     cov3d_precomp,
                                     signal_precomp,
                                     attenuation,
                                     raster_settings,
                                     )


class _TracerComplexGaussians(torch.autograd.Function):
    

    @staticmethod
    def forward(ctx,
                means_3d,
                cov3d_precomp,
                signal_precomp,
                attenuation,
                raster_settings
                ):
        radius_rx = raster_settings.radius_rx
        position = raster_settings.rx_pos

        scale_dis = 1.5
        radius_rx_filter = radius_rx * scale_dis
        distances        = torch.norm(means_3d - position, dim=1)

        indices_to_keep = (distances > radius_rx_filter)

        # Restructure arguments the way that the C++ lib expects them
        args = (means_3d[indices_to_keep],
                cov3d_precomp[indices_to_keep],
                signal_precomp[indices_to_keep],
                attenuation[indices_to_keep],
                raster_settings.gaus_radii[indices_to_keep],
                raster_settings.height,
                raster_settings.width,
                raster_settings.sh_degree_active,
                raster_settings.spectrum_3d_coarse,
                raster_settings.spectrum_3d_fine,
                raster_settings.rx_pos,
                raster_settings.radius_rx,
                raster_settings.tx_pos,
                raster_settings.bg, 
                raster_settings.debug
                )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:

            cpu_args = cpu_deep_copy_tuple(args)

            try:
                num_rendered, color, geomBuffer, binningBuffer, imgBuffer = _C.tracer_complex_gaussians(*args)

            except Exception as ex:

                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")

                raise ex
            
        else:

            num_rendered, color, geomBuffer, binningBuffer, imgBuffer = _C.tracer_complex_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.indices_to_keep = indices_to_keep
        ctx.save_for_backward(means_3d, cov3d_precomp, signal_precomp, attenuation,\
                              geomBuffer, binningBuffer, imgBuffer)
        
        return color


    @staticmethod
    def backward(ctx, 
                 grad_out_color
                 ):
        

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        indices_to_keep = ctx.indices_to_keep

        means_3d, cov3d_precomp, signal_precomp, attenuation,\
            geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (grad_out_color,
                means_3d[indices_to_keep], 
                cov3d_precomp[indices_to_keep],
                signal_precomp[indices_to_keep],
                attenuation[indices_to_keep],
                raster_settings.gaus_radii[indices_to_keep],
                num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                raster_settings.height,
                raster_settings.width,
                raster_settings.sh_degree_active,
                raster_settings.spectrum_3d_coarse,
                raster_settings.spectrum_3d_fine,
                raster_settings.rx_pos,
                raster_settings.radius_rx,
                raster_settings.tx_pos,
                raster_settings.bg, 
                raster_settings.debug
                )
        
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means_3d, grad_cov3d_precomp, \
                    grad_signal_precomp, grad_attenuation = _C.tracer_complex_gaussians_backward(*args)
            
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
            
        else:
            grad_means_3d, grad_cov3d_precomp, \
                grad_signal_precomp, grad_attenuation = _C.tracer_complex_gaussians_backward(*args)
            
        final_grad_means_3d = torch.zeros_like(means_3d)
        final_grad_means_3d[indices_to_keep] = grad_means_3d

        final_grad_cov3d_precomp = torch.zeros_like(cov3d_precomp)
        final_grad_cov3d_precomp[indices_to_keep] = grad_cov3d_precomp

        final_grad_signal_precomp = torch.zeros_like(signal_precomp)
        final_grad_signal_precomp[indices_to_keep] = grad_signal_precomp

        final_grad_attenuation = torch.zeros_like(attenuation)
        final_grad_attenuation[indices_to_keep] = grad_attenuation

        grads = (final_grad_means_3d,
                 final_grad_cov3d_precomp,
                 final_grad_signal_precomp,
                 final_grad_attenuation,
                 None
                 )

        return grads


class ComplexGaussianTracerSettings(NamedTuple):
    height             : int
    width              : int
    sh_degree_active   : int
    spectrum_3d_coarse : torch.Tensor
    spectrum_3d_fine   : torch.Tensor
    rx_pos             : torch.Tensor
    radius_rx          : float
    tx_pos             : torch.Tensor
    bg                 : torch.Tensor
    debug              : bool
    gaus_radii         : torch.Tensor


class ComplexGaussianTracer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings


    def forward(self, means_3d, cov3d_precomp, signal_precomp, attenuation):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return tracer_complex_gaussians(means_3d,
                                   cov3d_precomp,
                                   signal_precomp,
                                   attenuation,
                                   raster_settings
                                   )
    
    


