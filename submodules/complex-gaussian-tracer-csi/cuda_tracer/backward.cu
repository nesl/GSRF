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


#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) renderCUDA(const float* __restrict__ dL_dout_color,
																const float* __restrict__ means_3d,
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
																const float* __restrict__ geom_rgb,
																const uint32_t* __restrict__ bin_point_list,
																const uint2* __restrict__ img_ranges,
																const float* __restrict__ final_Ts,
																const uint32_t* __restrict__ img_n_contrib,
																float* __restrict__ grad_means_3d,
																float* __restrict__ grad_cov3d_precomp,
																float* __restrict__ grad_attenuation,
																float* __restrict__ dL_dcolors,
																bool debug
																)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	const uint2 pix_min = { block.group_index().x * BLOCK_X,
							block.group_index().y * BLOCK_Y };

	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W),
							min(pix_min.y + BLOCK_Y , H) };

	const uint2 pix = { pix_min.x + block.thread_index().x,
						pix_min.y + block.thread_index().y };

	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x,
						  (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside
	const bool inside = ((pix.x < W) && (pix.y < H));

	// 	Load start/end range of IDs to process in bit sorted list
	const uint2 range = img_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ glm::vec3 collected_means_3d[BLOCK_SIZE];
	__shared__ glm::mat3 collected_cov3d[BLOCK_SIZE];
	__shared__ float    collected_attenuation[BLOCK_SIZE];
	// Signal (52 channels) read from global memory


	glm::vec3 pix_pos;

	if (inside) {
		pix_pos = glm::vec3( spectrum_3d_fine[3 * pix_id + 0],
							 spectrum_3d_fine[3 * pix_id + 1],
							 spectrum_3d_fine[3 * pix_id + 2] );
	} else {

		pix_pos = glm::vec3( -1.0f, -1.0f, -1.0f );
	}

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? img_n_contrib[pix_id] : 0;

	float last_alpha = 0.0f;
	float accum_rec[C] = { 0 };
	float last_sig[C] = { 0 };

	float dL_dpixel[C] = { 0 };
	if (inside) {
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dout_color[i * H * W + pix_id];
		}
	}

	const float const_atten = 0.99f;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {

		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {

			const int gau_idx = bin_point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = gau_idx;

			collected_means_3d[block.thread_rank()] = glm::vec3( means_3d[3 * gau_idx + 0],
																 means_3d[3 * gau_idx + 1],
																 means_3d[3 * gau_idx + 2] );

			const float* cov3d = cov3d_precomp + 6 * gau_idx;
			collected_cov3d[block.thread_rank()] = glm::inverse(glm::mat3( cov3d[0], cov3d[1], cov3d[2],
																		   cov3d[1], cov3d[3], cov3d[4],
																		   cov3d[2], cov3d[4], cov3d[5] ));

			collected_attenuation[block.thread_rank()] = attenuation[gau_idx];

		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			contributor--;
			if (contributor >= last_contributor) {
				continue;
			}

			int gau_idx = collected_id[j];
			glm::vec3 gau_pos = collected_means_3d[j];
			glm::mat3 covMatrix = collected_cov3d[j];
			float gau_atten = collected_attenuation[j];

			// Read signal from global memory
			const float* sig = geom_rgb + C * gau_idx;

			float power = calculate_exponent(pix_pos,
											 gau_pos,
											 *sphere_center,
											 sphere_radius,
											 covMatrix
											 );

			if (power > 0.0f) {
				continue;
			}

			const float G = exp(power);
			const float temp_alpha = gau_atten * G;
			const float alpha = min(const_atten, temp_alpha);
			if (alpha < 1.0f / 255.0f) {
				continue;
			}

			T = T / (1.f - alpha);

			// Signal gradients for all C channels
			for (int c = 0; c < C; c++) {
				float dL_dsig_c = alpha * T * dL_dpixel[c];
				atomicAdd(&(dL_dcolors[C * gau_idx + c]), dL_dsig_c);
			}

			// Accumulated signal for alpha gradient (sum over all channels)
			float dL_dalpha = 0.0f;
			for (int c = 0; c < C; c++) {
				accum_rec[c] = last_alpha * last_sig[c] + (1.0f - last_alpha) * accum_rec[c];
				dL_dalpha += (sig[c] - accum_rec[c]) * dL_dpixel[c];
			}
			dL_dalpha *= T;

			for (int c = 0; c < C; c++) {
				last_sig[c] = sig[c];
			}
			last_alpha = alpha;

			float dL_dG;
			float dL_datten;

			if (temp_alpha <= const_atten) {
				dL_datten = dL_dalpha * G;
				dL_dG     = dL_dalpha * gau_atten;
			} else {
				dL_datten = 0.0f;
				dL_dG     = 0.0f;
			}

			atomicAdd(&(grad_attenuation[gau_idx]), dL_datten);

			float dL_dpower = dL_dG * G;

			glm::vec3 dL_dgau_pos;
			glm::mat3 dL_dinvCovMatrix;
			calculate_exponent_grad_v3(pix_pos,
									   gau_pos,
									   *sphere_center,
									   sphere_radius,
									   covMatrix,
									   dL_dpower,
									   dL_dgau_pos,
									   dL_dinvCovMatrix);

			atomicAdd(&(grad_means_3d[3 * gau_idx + 0]), dL_dgau_pos.x);
			atomicAdd(&(grad_means_3d[3 * gau_idx + 1]), dL_dgau_pos.y);
			atomicAdd(&(grad_means_3d[3 * gau_idx + 2]), dL_dgau_pos.z);

			float dL_dcov3d[6]{};
			calculateGradientWrtCov3d(dL_dinvCovMatrix,
									  covMatrix,
									  dL_dcov3d
									  );

			for (int i = 0; i < 6; ++i) {
				atomicAdd(&(grad_cov3d_precomp[6 * gau_idx + i]), dL_dcov3d[i]);
			}

		}
	}

}


void BACKWARD::render(const dim3 grid,
					  const dim3 block,
					  const float* dL_dout_color,
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
					  const float* geom_rgb,
					  const uint32_t* bin_point_list,
					  const uint2* img_ranges,
					  const float* final_Ts,
					  const uint32_t* img_n_contrib,
					  float* grad_means_3d,
					  float* grad_cov3d_precomp,
					  float* grad_attenuation,
					  float* dL_dcolors,
					  bool debug)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(dL_dout_color,
												  means_3d,
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
												  geom_rgb,
												  bin_point_list,
												  img_ranges,
												  final_Ts,
												  img_n_contrib,
												  grad_means_3d,
												  grad_cov3d_precomp,
												  grad_attenuation,
												  dL_dcolors,
												  debug
												  );

}

