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


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
							   const float* means_3d,
							   const float* cov3d_precomp,
							   const float* signal_precomp,
							   const float* gaus_radii,
							   const int H,
							   const int W,
							   const glm::vec3* sphere_center,
							   const float sphere_radius,
							   const int fle_degree_active,
							   const int fle_coef_len_max,
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

	float dist = glm::distance(*sphere_center, p_orig);  
	const float scale_dis = 1.5f; 
	if (dist < sphere_radius * scale_dis) {
		return;
	}

	float my_radius = gaus_radii[idx];

	const float* cov3d = cov3d_precomp + 6 * idx;

	float2 point_image;
	cartesian_to_spherical(p_orig, *sphere_center, cov3d, point_image);

	if (point_image.y <= 0.0f) {
		return;
	}

	float angle_radius;
    calculate_central_angle(my_radius, sphere_radius, angle_radius);

	uint2 rect_min, rect_max;
	getRect_v3(point_image, angle_radius, rect_min, rect_max, tile_grid);
	if ((rect_max.x < rect_min.x) || (rect_max.y < rect_min.y)) {
		return;
	}

	const float* color_pt = signal_precomp + C * idx;
	for (int c = 0; c < C; c++) {
		geom_rgb[C * idx + c] = color_pt[c];
	}

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
																const int fle_degree_active,
																const int fle_coef_len_max,
																const float* __restrict__ spectrum_3d_fine,
																const glm::vec3* sphere_center,
																const float sphere_radius,
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
	// Note: signal (52 channels) too large for shared memory — read from global

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

	float receive_signal[CHANNELS] = { 0 };

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

		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {

			contributor++;

			int gau_idx = collected_id[j];
			glm::vec3 gau_pos = collected_means_3d[j];
			glm::mat3 covMatrix = collected_cov3d[j];
			float gau_atten = collected_attenuation[j];

			float power = calculate_exponent(pix_pos,
											 gau_pos,
											 *sphere_center,
											 sphere_radius,
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

			// Alpha-composite all 52 channels from global memory
			const float* sig = geom_rgb + CHANNELS * gau_idx;
			for (int c = 0; c < CHANNELS; c++) {
				receive_signal[c] += alpha * T * sig[c];
			}

			T = test_T;

			last_contributor = contributor;

		}
	}

	if (inside) {

		img_accum_alpha[pix_id] = T;

		img_n_contrib[pix_id] = last_contributor;

		for (int c = 0; c < CHANNELS; c++) {
			out_color[c * H * W + pix_id] = receive_signal[c];
		}

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
					 const int fle_degree_active,
					 const int fle_coef_len_max,
					 const float* spectrum_3d_fine,
					 const glm::vec3* sphere_center,
					 const float sphere_radius,
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
														fle_degree_active,
														fle_coef_len_max,
														spectrum_3d_fine,
														sphere_center,
														sphere_radius,
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
						 const glm::vec3* sphere_center,
						 const float sphere_radius,
						 const int fle_degree_active,
						 const int fle_coef_len_max,
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
																sphere_center,
																sphere_radius,
																fle_degree_active,
																fle_coef_len_max,
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


