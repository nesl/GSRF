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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_tracer/config.h"
#include "cuda_tracer/tracer.h"
#include <fstream>
#include <string>
#include <functional>


// defines a function resizeFunctional that creates and returns a lambda function (closure) 
// 		to resize a given PyTorch tensor and return a pointer to its data.

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {

    auto lambda = [&t](size_t N) {

        t.resize_({(long long)N});

		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };

    return lambda;
}


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
					   )
{

	if (means_3d.ndimension() != 2 || means_3d.size(1) != 3) {
		AT_ERROR("means 3D must have dimensions (num_points, 3)");
	}

	const int P = means_3d.size(0);
	const int H = height;
	const int W = width;

	// printf("\n\n\nNumber of points in Forward: %d\\nn", P);

	auto int_opts   = means_3d.options().dtype(torch::kInt32);
	auto float_opts = means_3d.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);

	torch::Tensor geomBuffer     = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer  = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer      = torch::empty({0}, options.device(device));

	std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

	int rendered = 0;
	if(P != 0) {

		const int M = 16;

		rendered = CudaTracer::Tracer::forward(P,
												means_3d.contiguous().data_ptr<float>(),
												cov3d_precomp.contiguous().data_ptr<float>(),
												signal_precomp.contiguous().data_ptr<float>(),
												attenuation.contiguous().data_ptr<float>(),
												gaus_radii.contiguous().data_ptr<float>(),
												H,
												W,
												sh_degree_active,
												M,
												spectrum_3d_coarse.contiguous().data_ptr<float>(),
												spectrum_3d_fine.contiguous().data_ptr<float>(),
												rx_pos.contiguous().data_ptr<float>(),
												radius_rx,
												tx_pos.contiguous().data_ptr<float>(),
												background.contiguous().data_ptr<float>(),
												out_color.contiguous().data_ptr<float>(),
												geomFunc,
												binningFunc,
												imgFunc,
												debug
												);
	
	}


	return std::make_tuple(rendered, out_color, geomBuffer, binningBuffer, imgBuffer);

}


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
							   const torch::Tensor& spectrum_3d_coarse,
							   const torch::Tensor& spectrum_3d_fine,
							   const torch::Tensor& rx_pos,
							   const float radius_rx,
							   const torch::Tensor& tx_pos,
							   const torch::Tensor& background,
							   const bool debug
							   )
{

	const int P = means_3d.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	// printf("\n\n\nNumber of points in Backward: %d\\nn", P);

	const int M = 16;

	torch::Tensor grad_means_3d        = torch::zeros({P, 3},    means_3d.options());
	torch::Tensor grad_cov3d_precomp   = torch::zeros({P, 6},    means_3d.options());

	torch::Tensor grad_attenuation     = torch::zeros({P, 1},    means_3d.options());


	torch::Tensor dL_dcolors           = torch::zeros({P, NUM_CHANNELS}, means_3d.options());

	if(P != 0) {

		CudaTracer::Tracer::backward(P,
										dL_dout_color.contiguous().data_ptr<float>(),
										means_3d.contiguous().data_ptr<float>(),
										cov3d_precomp.contiguous().data_ptr<float>(),
										signal_precomp.contiguous().data_ptr<float>(),
										attenuation.contiguous().data_ptr<float>(),
										gaus_radii.contiguous().data_ptr<float>(),
										num_rendered,
										reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
										reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
										reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
										H,
										W,
										sh_degree_active,
										M,
										spectrum_3d_coarse.contiguous().data_ptr<float>(),
										spectrum_3d_fine.contiguous().data_ptr<float>(),
										rx_pos.contiguous().data_ptr<float>(),
										radius_rx,
										tx_pos.contiguous().data_ptr<float>(),
										background.contiguous().data_ptr<float>(),
										dL_dcolors.contiguous().data_ptr<float>(),
										grad_means_3d.contiguous().data_ptr<float>(),
										grad_cov3d_precomp.contiguous().data_ptr<float>(),
										grad_attenuation.contiguous().data_ptr<float>(),
										debug
										);
	}

	return std::make_tuple(grad_means_3d, grad_cov3d_precomp, dL_dcolors, grad_attenuation);


}




