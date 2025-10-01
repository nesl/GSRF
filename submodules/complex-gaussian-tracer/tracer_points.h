/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */


#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TracerComplexGaussiansCUDA(const torch::Tensor& means_3d,
					   const torch::Tensor& cov3d_precomp,
					   const torch::Tensor& signal_precomp,
					   const torch::Tensor& attenuation,
					   const torch::Tensor& gaus_radii,
					   const int height,
					   const int width,
					   const int sh_degree_active,
					   // const int sh_coef_len_max,
					   const torch::Tensor& spectrum_3d_coarse,
					   const torch::Tensor& spectrum_3d_fine,
					   const torch::Tensor& rx_pos,
					   const float radius_rx, 
					   const torch::Tensor& tx_pos,
					   const torch::Tensor& background,
					//    const int input_dim,
					//    const int hidden_dim,
					//    const int output_dim, 
					//    const int total_params,
					   const bool debug
					   );


// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
TracerComplexGaussiansBackwardCUDA(const torch::Tensor& dL_dout_color,
							   const torch::Tensor& means_3d,
							   const torch::Tensor& cov3d_precomp,
							   const torch::Tensor& signal_precomp,
							   const torch::Tensor& attenuation,
							   const torch::Tensor& gaus_radii,
							   const int num_rendered,
							   const torch::Tensor& geomBuffer,
							   const torch::Tensor& binningBuffer,
							   const torch::Tensor& imageBuffer,
							   const int height,
							   const int width,
							   const int sh_degree_active,
							   // const int sh_coef_len_max,
							   const torch::Tensor& spectrum_3d_coarse,
							   const torch::Tensor& spectrum_3d_fine,
							   const torch::Tensor& rx_pos,
							   const float radius_rx,
							   const torch::Tensor& tx_pos,
							   const torch::Tensor& background,
							//    const int input_dim,
							//    const int hidden_dim,
							//    const int output_dim, 
							//    const int total_params,
							   const bool debug
							   );
		



