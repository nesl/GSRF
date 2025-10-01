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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuComplex.h>

#include <cstdio>

#include <curand_kernel.h>
#include <stdio.h>

namespace FORWARD
{
	
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, 
					const float* means_3d,
					const float* cov3d_precomp,
					const float* signal_precomp,
					const float* gaus_radii,
					const int H, 
					const int W,
					const float* spectrum_3d_coarse,
					const glm::vec3* rx_pos,
					const float radius_rx,
					const float* tx_pos,
					const int sh_degree_active,
					const int sh_coef_len_max, 
					float* geom_depths,
					uint32_t* geom_tiles_touched,
					uint2* geom_rec_mins,
					uint2* geom_rec_maxs,
					float2* geom_means_2d,
					float* geom_rgb,
					bool* geom_clamped,
					const dim3 tile_grid
					);


	// Main rendering method.
	void render(const dim3 tile_grid, 
				const dim3 block,
				const float* means_3d,
				const float* cov3d_precomp,
				const float* attenuation,
				const float* gaus_radii,
				const int H,
				const int W,
				const int sh_degree_active,
				const int sh_coef_len_max,
				const float* spectrum_3d_fine,
				const glm::vec3* rx_pos,
				const float radius_rx,
				const float* bg_color,
				const uint32_t* bin_point_list,
				const uint2* img_ranges,
				const float2* geom_means_2d,
				const float* geom_rgb,
				float* img_accum_alpha,  
				uint32_t* img_n_contrib,
				float* out_color
				);

}


#endif