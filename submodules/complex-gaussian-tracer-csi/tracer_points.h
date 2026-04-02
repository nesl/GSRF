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
					   const int fle_degree_active,
					   const torch::Tensor& spectrum_3d_fine,
					   const torch::Tensor& sphere_center,
					   const float sphere_radius,
					   const bool debug
					   );


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
							   const int fle_degree_active,
							   const torch::Tensor& spectrum_3d_fine,
							   const torch::Tensor& sphere_center,
							   const float sphere_radius,
							   const bool debug
							   );
		



