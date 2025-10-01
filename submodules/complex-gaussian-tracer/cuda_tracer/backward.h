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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <cuComplex.h>

namespace BACKWARD
{
	
	void render(const dim3 grid, 
				const dim3 block,
				const float* dL_dout_color,
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
				const float* geom_rgb,
				const uint32_t* bin_point_list,
				const uint2* img_ranges,
				const float* final_Ts,
				const uint32_t* img_n_contrib,
				float* grad_means_3d,
				float* grad_cov3d_precomp,
				float* grad_attenuation,
				float* dL_dcolors,
				bool debug
				);

	void preprocess(const int P,
					const float3* means_3d,
					const float* signal_precomp,
					const glm::vec3* rx_pos,
					const float radius_rx,
					const float* tx_pos,
					const int sh_degree_active,
					const int sh_coef_len_max, 
					bool* geom_clamped,
					float* dL_dcolor,
					float* grad_means_3d,
					float* grad_signal_precomp,
					bool debug);

}

#endif