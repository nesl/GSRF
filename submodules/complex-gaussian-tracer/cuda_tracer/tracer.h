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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuComplex.h>


namespace CudaTracer
{
	class Tracer
	{
		public:

			static int forward(const int P,
							   const float* means_3d,
							   const float* cov3d_precomp,
							   const float* signal_precomp,
							   const float* attenuation,
							   const float* gaus_radii,
							   const int H,
							   const int W,
							   const int sh_degree_active,
							   const int sh_coef_len_max,
							   const float* spectrum_3d_coarse,
							   const float* spectrum_3d_fine,
							   const float* rx_pos,
							   const float radius_rx,
							   const float* tx_pos,
							   const float* background,
							   float* out_color,
							   std::function<char* (size_t)> geometryBuffer,
							   std::function<char* (size_t)> binningBuffer,
							   std::function<char* (size_t)> imageBuffer,
							   bool debug
							   );


			static void backward(const int P,
								 const float* dL_dout_color,
								 const float* means_3d,
								 const float* cov3d_precomp,
								 const float* signal_precomp,
								 const float* attenuation,
								 const float* gaus_radii,
								 const int num_rendered,
								 char* geom_buffer,
								 char* binning_buffer,
								 char* image_buffer,
								 const int H,
								 const int W,
								 const int sh_degree_active,
								 const int sh_coef_len_max,
								 const float* spectrum_3d_coarse,
								 const float* spectrum_3d_fine,
								 const float* rx_pos,
								 const float radius_rx,
								 const float* tx_pos,
								 const float* background,
								 float* dL_dcolor,
								 float* grad_means_3d,
								 float* grad_cov3d_precomp,
								 float* grad_attenuation,
								 bool debug
								 );
			
	};
};

#endif

