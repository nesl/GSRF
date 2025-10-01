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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <glm/glm.hpp>

namespace cg = cooperative_groups;


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, 
										int deg, 
										int max_coeffs, 
										const glm::vec3* means, 
										glm::vec3 campos, 
										const float* shs, 
										bool* clamped
										)
{

	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result 
					- SH_C1 * y * sh[1]
					+ SH_C1 * z * sh[2]
					- SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			result = result 
						+ SH_C2[0] * xy                    * sh[4]
						+ SH_C2[1] * yz                    * sh[5]
						+ SH_C2[2] * (2.0f * zz - xx - yy) * sh[6]
						+ SH_C2[3] * xz                    * sh[7] 
						+ SH_C2[4] * (xx - yy)             * sh[8];

			if (deg > 2)
			{
				result = result
							+ SH_C3[0] * y  * (3.0f * xx - yy)                    * sh[9]
							+ SH_C3[1] * xy * z                                   * sh[10] 
							+ SH_C3[2] * y  * (4.0f * zz - xx - yy)               * sh[11]
							+ SH_C3[3] * z  * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12]
							+ SH_C3[4] * x  * (4.0f * zz - xx - yy)               * sh[13] 
							+ SH_C3[5] * z  * (xx - yy)                           * sh[14]
							+ SH_C3[6] * x  * (xx - 3.0f * yy)                    * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);

	// clamped[3 * idx + 0] = false;
	// clamped[3 * idx + 1] = false;
	// clamped[3 * idx + 2] = false;

	return glm::max(result, 0.0f);
	// return result;


}


__device__ glm::vec3 computeColorFromMLP_v2(int idx, 
											const glm::vec3* means, 
											glm::vec3 campos, 
											const float* all_parameters, 
											const float* tx_pos
											)
{

    // Compute the direction vector
	glm::vec3 pos = means[idx];
    glm::vec3 direction_vector = pos - campos;
	float dir_length = glm::length(direction_vector);

    glm::vec3 dir = direction_vector / dir_length;

	float dir_input[INPUT_DIM_DIR] = {dir.x, dir.y, dir.z};
	// int embedding_dim = INPUT_DIM_DIR + 2 * INPUT_DIM_DIR * NUM_FREQS;

	float dir_embedding[EMBEDDING_DIM];
	create_embedding_fn(dir_input, dir_embedding);

	// INPUT_DIM_EMD = EMBEDDING_DIM * 2
	float vector_dir_pos[INPUT_DIM_EMD];

	for (int i = 0; i < EMBEDDING_DIM; ++i) {
        vector_dir_pos[i] = dir_embedding[i];
    }

    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        vector_dir_pos[EMBEDDING_DIM + i] = tx_pos[i];
    }

    // Pointer to the parameters for the current index
    const float* parameter_mlp = all_parameters + TOTAL_PARAMS * idx;

    // Extract weights and biases for the first layer
    const float* weights_1 = parameter_mlp;
    const float* bias_1    = weights_1 + (INPUT_DIM_EMD * HIDDEN_DIM_1);

    // Extract weights and biases for the second layer
    const float* weights_2 = bias_1 + HIDDEN_DIM_1;
    const float* bias_2    = weights_2 + (HIDDEN_DIM_1 * HIDDEN_DIM_2);

    // Extract weights and biases for the third layer
    const float* weights_3 = bias_2 + HIDDEN_DIM_2;
    const float* bias_3    = weights_3 + (HIDDEN_DIM_2 * OUTPUT_DIM);

    // First layer computation
    float hidden_1[HIDDEN_DIM_1];
    for (int i = 0; i < HIDDEN_DIM_1; ++i) {
        hidden_1[i] = bias_1[i];

        for (int j = 0; j < INPUT_DIM_EMD; ++j) {
            hidden_1[i] += vector_dir_pos[j] * weights_1[j * HIDDEN_DIM_1 + i];
        }

        hidden_1[i] = leaky_relu(hidden_1[i]);
    }

    // Second layer computation
    float hidden_2[HIDDEN_DIM_2];
    for (int i = 0; i < HIDDEN_DIM_2; ++i) {
        hidden_2[i] = bias_2[i];

        for (int j = 0; j < HIDDEN_DIM_1; ++j) {
            hidden_2[i] += hidden_1[j] * weights_2[j * HIDDEN_DIM_2 + i];
        }

        hidden_2[i] = leaky_relu(hidden_2[i]);
    }

    // Third layer computation
    float output[OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        output[i] = bias_3[i];

        for (int j = 0; j < HIDDEN_DIM_2; ++j) {
            output[i] += hidden_2[j] * weights_3[j * OUTPUT_DIM + i];
        }

        output[i] = sigmoid(output[i]);
    }

    // Convert float array to glm::vec3
    glm::vec3 output_vec = glm::vec3(output[0], output[1], output[2]);

    return output_vec;
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
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
							   )
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	geom_tiles_touched[idx] = 0;

	glm::vec3 p_orig = glm::vec3( means_3d[3 * idx + 0],
								  means_3d[3 * idx + 1],
								  means_3d[3 * idx + 2] );

	float dist = glm::distance(*rx_pos, p_orig);  
	const float scale_dis = 1.5f; 
	if (dist < radius_rx * scale_dis) {
		return;
	}

	float my_radius = gaus_radii[idx];

	const float* cov3d = cov3d_precomp + 6 * idx;

	float2 point_image;
	cartesian_to_spherical(p_orig, *rx_pos, cov3d, point_image);

	if (point_image.y >= 90.0f) {
		return;
	}

	float angle_radius;
    calculate_central_angle(my_radius, radius_rx, angle_radius);

	uint2 rect_min, rect_max;
	getRect_v3(point_image, my_radius, rect_min, rect_max, tile_grid);
	if ((rect_max.x < rect_min.x) || (rect_max.y < rect_min.y)) {
		return;
	}

	const float* color_pt = signal_precomp + C * idx;
	geom_rgb[C * idx + 0] = color_pt[0];
	geom_rgb[C * idx + 1] = color_pt[1];

	geom_depths[idx] = dist;

	geom_means_2d[idx] = point_image;

	geom_tiles_touched[idx] = (rect_max.x - rect_min.x + 1) * (rect_max.y - rect_min.y + 1);

	geom_rec_mins[idx] = rect_min;

	geom_rec_maxs[idx] = rect_max;

}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) renderCUDA(const float* __restrict__ means_3d,
																const float* __restrict__ cov3d_precomp,
																const float* __restrict__ attenuation,
																const float* __restrict__ gaus_radii,
																const int H,
																const int W,
																const int sh_degree_active,
																const int sh_coef_len_max,
																const float* __restrict__ spectrum_3d_fine,
																const glm::vec3* rx_pos,
																const float radius_rx,
																const float* __restrict__ bg_color,
																const uint32_t* __restrict__ bin_point_list,
																const uint2* __restrict__ img_ranges,
																const float2* __restrict__ geom_means_2d,
																const float* geom_rgb,
																float* __restrict__ img_accum_alpha,
																uint32_t* __restrict__ img_n_contrib,
																float* __restrict__ out_color
																)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	uint2 pix_min = { block.group_index().x * BLOCK_X,
					  block.group_index().y * BLOCK_Y };

	uint2 pix_max = { min(pix_min.x + BLOCK_X, W),
					  min(pix_min.y + BLOCK_Y, H) };

	uint2 pix = { pix_min.x + block.thread_index().x,
				  pix_min.y + block.thread_index().y };

	uint32_t pix_id = W * pix.y + pix.x;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = ((pix.x < W) && (pix.y < H));

	bool done = !inside;

	uint2 range = img_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];



	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo         = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ glm::vec3 collected_means_3d[BLOCK_SIZE];
	__shared__ glm::mat3 collected_cov3d[BLOCK_SIZE];
	__shared__ float    collected_attenuation[BLOCK_SIZE];
	__shared__ float2 collected_signal[BLOCK_SIZE];

	glm::vec3 pix_pos;

	if (inside) {

		pix_pos = glm::vec3( spectrum_3d_fine[3 * pix_id + 0],
							 spectrum_3d_fine[3 * pix_id + 1],
							 spectrum_3d_fine[3 * pix_id + 2] );
	} else {

		pix_pos = glm::vec3( -1.0f, -1.0f, -1.0f );
	}


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;

	cuFloatComplex receive_signal = make_cuFloatComplex(0.0f, 0.0f);

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {

		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {

			int gau_idx = bin_point_list[range.x + progress];

			collected_id[block.thread_rank()] = gau_idx;

			collected_means_3d[block.thread_rank()] = glm::vec3(means_3d[3 * gau_idx + 0],
																means_3d[3 * gau_idx + 1],
																means_3d[3 * gau_idx + 2]);

			const float* cov3d = cov3d_precomp + 6 * gau_idx;
			collected_cov3d[block.thread_rank()] = glm::inverse(glm::mat3( cov3d[0], cov3d[1], cov3d[2], 
																		   cov3d[1], cov3d[3], cov3d[4], 
																		   cov3d[2], cov3d[4], cov3d[5] )
																);

			collected_attenuation[block.thread_rank()] = attenuation[gau_idx];

			collected_signal[block.thread_rank()] = make_float2(geom_rgb[2 * gau_idx + 0], 
																geom_rgb[2 * gau_idx + 1]);

		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {

			contributor++;

			int gau_idx = collected_id[j];
			glm::vec3 gau_pos = collected_means_3d[j];
			glm::mat3 covMatrix = collected_cov3d[j];
			float gau_atten = collected_attenuation[j];

			float2 gau_signal = collected_signal[j];
			float sig_amp = gau_signal.x;
			float sig_pha = gau_signal.y;

			float power = calculate_exponent(pix_pos,
											 gau_pos,
											 *rx_pos,
											 radius_rx,
											 covMatrix
											 );

			if (power > 0.0f) {
				continue;
			}

			float alpha = min(0.99f, gau_atten * exp(power));

			if (alpha < 1.0f / 255.0f) {
				continue;
			}

			float test_T = T * (1 - alpha);

			if (test_T < 0.0001f) {
				done = true;
				continue;
			}

			cuFloatComplex sig_cur   = complexExpMult(sig_amp, sig_pha);
			cuFloatComplex alpha_cur = make_cuFloatComplex(alpha, 0.0f);
			cuFloatComplex T_cur     = make_cuFloatComplex(T, 0.0f);

			cuFloatComplex product  = cuCmulf(alpha_cur, T_cur);
			cuFloatComplex sig_temp = cuCmulf(product, sig_cur);
			receive_signal = cuCaddf(receive_signal, sig_temp);

			T = test_T;

			last_contributor = contributor;

		}
	}

	if (inside) {

		img_accum_alpha[pix_id] = T;

		img_n_contrib[pix_id] = last_contributor;

		out_color[0 * H * W + pix_id] = receive_signal.x;
		out_color[1 * H * W + pix_id] = receive_signal.y;
		
	}

}


void FORWARD::render(const dim3 tile_grid, 
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
					 )
{
	renderCUDA<NUM_CHANNELS> << <tile_grid, block >> > (means_3d,
														cov3d_precomp,
														attenuation,
														gaus_radii,
														H,
														W,
														sh_degree_active,
														sh_coef_len_max,
														spectrum_3d_fine,
														rx_pos,
														radius_rx,
														bg_color,
														bin_point_list,
														img_ranges,
														geom_means_2d,
														geom_rgb,
														img_accum_alpha,  
														img_n_contrib,
														out_color
														);

}


void FORWARD::preprocess(int P,
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
						 )
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (P,
																means_3d,
																cov3d_precomp,
																signal_precomp,
																gaus_radii,
																H,
																W,
																spectrum_3d_coarse,
																rx_pos,
																radius_rx,
																tx_pos,
																sh_degree_active,
																sh_coef_len_max, 
																geom_depths,
																geom_tiles_touched,
																geom_rec_mins,
																geom_rec_maxs,
																geom_means_2d,
																geom_rgb,
																geom_clamped,
																tile_grid
																);


}


