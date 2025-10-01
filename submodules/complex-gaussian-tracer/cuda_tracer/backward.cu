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

#include <cmath>


__device__ void computeColorFromMLP_gradients(int idx, 
                                              const glm::vec3* means, 
                                              const glm::vec3 campos, 
                                              const float* all_parameters, 
											  const float* tx_pos,
                                              const glm::vec3* dL_doutput_vec, 
                                              float* dL_dparas,
                                              glm::vec3* dL_dmeans
											  ) 
{
    // Compute the direction vector
    glm::vec3 pos = means[idx];
    glm::vec3 direction_vector = pos - campos;
	float dir_length = glm::length(direction_vector);
    glm::vec3 dir = direction_vector / dir_length;

    float dir_input[INPUT_DIM_DIR] = {dir.x, dir.y, dir.z};

    float dir_embedding[EMBEDDING_DIM];
    create_embedding_fn(dir_input, dir_embedding);

    float vector_dir_pos[INPUT_DIM_EMD];
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        vector_dir_pos[i] = dir_embedding[i];
    }
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        vector_dir_pos[EMBEDDING_DIM + i] = tx_pos[i];
    }

    const float* parameter_mlp = all_parameters + TOTAL_PARAMS * idx;
    float* dL_dparas_one_gaus  = dL_dparas + TOTAL_PARAMS * idx;

    const float* weights_1 = parameter_mlp;
    const float* bias_1    = weights_1 + (INPUT_DIM_EMD * HIDDEN_DIM_1);
    const float* weights_2 = bias_1    + HIDDEN_DIM_1;
    const float* bias_2    = weights_2 + (HIDDEN_DIM_1 * HIDDEN_DIM_2);
    const float* weights_3 = bias_2    + HIDDEN_DIM_2;
    const float* bias_3    = weights_3 + (HIDDEN_DIM_2 * OUTPUT_DIM);

    float hidden_1[HIDDEN_DIM_1];
    for (int i = 0; i < HIDDEN_DIM_1; ++i) {
        hidden_1[i] = bias_1[i];
        for (int j = 0; j < INPUT_DIM_EMD; ++j) {
            hidden_1[i] += vector_dir_pos[j] * weights_1[j * HIDDEN_DIM_1 + i];
        }
        hidden_1[i] = leaky_relu(hidden_1[i]);
    }

    float hidden_2[HIDDEN_DIM_2];
    for (int i = 0; i < HIDDEN_DIM_2; ++i) {
        hidden_2[i] = bias_2[i];
        for (int j = 0; j < HIDDEN_DIM_1; ++j) {
            hidden_2[i] += hidden_1[j] * weights_2[j * HIDDEN_DIM_2 + i];
        }
        hidden_2[i] = leaky_relu(hidden_2[i]);
    }

    float output[OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        output[i] = bias_3[i];
        for (int j = 0; j < HIDDEN_DIM_2; ++j) {
            output[i] += hidden_2[j] * weights_3[j * OUTPUT_DIM + i];
        }
        output[i] = sigmoid(output[i]);
    }

    float dL_doutput[OUTPUT_DIM] = {dL_doutput_vec->x, dL_doutput_vec->y, dL_doutput_vec->z};

    float dL_dhidden_2[HIDDEN_DIM_2] = {0.0f};
    float dL_dweights_3[HIDDEN_DIM_2 * OUTPUT_DIM] = {0.0f};
    float dL_dbias_3[OUTPUT_DIM] = {0.0f};
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        float gradient = dL_doutput[i] * sigmoid_derivative(output[i]);
        dL_dbias_3[i] = gradient;
        for (int j = 0; j < HIDDEN_DIM_2; ++j) {
            dL_dweights_3[j * OUTPUT_DIM + i] = gradient * hidden_2[j];
            dL_dhidden_2[j] += gradient * weights_3[j * OUTPUT_DIM + i];
        }
    }

    float dL_dhidden_1[HIDDEN_DIM_1] = {0.0f};
    float dL_dweights_2[HIDDEN_DIM_1 * HIDDEN_DIM_2] = {0.0f};
    float dL_dbias_2[HIDDEN_DIM_2] = {0.0f};
    for (int i = 0; i < HIDDEN_DIM_2; ++i) {
        float gradient = dL_dhidden_2[i] * leaky_relu_derivative(hidden_2[i]);
        dL_dbias_2[i] = gradient;
        for (int j = 0; j < HIDDEN_DIM_1; ++j) {
            dL_dweights_2[j * HIDDEN_DIM_2 + i] = gradient * hidden_1[j];
            dL_dhidden_1[j] += gradient * weights_2[j * HIDDEN_DIM_2 + i];
        }
    }

    float dL_dvector_dir_pos[INPUT_DIM_EMD] = {0.0f};
    float dL_dweights_1[INPUT_DIM_EMD * HIDDEN_DIM_1] = {0.0f};
    float dL_dbias_1[HIDDEN_DIM_1] = {0.0f};
    for (int i = 0; i < HIDDEN_DIM_1; ++i) {
        float gradient = dL_dhidden_1[i] * leaky_relu_derivative(hidden_1[i]);
        dL_dbias_1[i] = gradient;
        for (int j = 0; j < INPUT_DIM_EMD; ++j) {
            dL_dweights_1[j * HIDDEN_DIM_1 + i] = gradient * vector_dir_pos[j];
            dL_dvector_dir_pos[j] += gradient * weights_1[j * HIDDEN_DIM_1 + i];
        }
    }

    float dL_ddir_embedding[EMBEDDING_DIM];
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        dL_ddir_embedding[i] = dL_dvector_dir_pos[i];
    }

    float dL_ddir_input[INPUT_DIM_DIR] = {0.0f};
    int out_idx = 0;
    float max_freq = (float)MAX_FREQ_LOG2;
    int N_freqs = NUM_FREQS;

    for (int i = 0; i < INPUT_DIM_DIR; ++i) {
        dL_ddir_input[i] = dL_ddir_embedding[i];
    }
    out_idx += INPUT_DIM_DIR;

    for (int i = 0; i < N_freqs; ++i) {
        float freq = powf(2.0f, i * max_freq / (N_freqs - 1));

        for (int j = 0; j < INPUT_DIM_DIR; ++j) {
            float sin_grad = dL_ddir_embedding[out_idx + j] * sin_derivative(dir_input[j] * freq);
            float cos_grad = dL_ddir_embedding[out_idx + INPUT_DIM_DIR + j] * cos_derivative(dir_input[j] * freq);
            dL_ddir_input[j] += freq * (sin_grad + cos_grad);
        }

        out_idx += 2 * INPUT_DIM_DIR;
    }

	glm::mat3 identity_matrix = glm::mat3(1.0f); 
    glm::mat3 outer_norm = glm::outerProduct(direction_vector, direction_vector);
    glm::mat3 d_norm_dir_d_direction_vector = (1.0f / dir_length) * (identity_matrix - outer_norm / (dir_length * dir_length));

    glm::vec3 grad_dir = glm::vec3(dL_ddir_input[0], dL_ddir_input[1], dL_ddir_input[2]);

    glm::vec3 dL_ddirection_vector = d_norm_dir_d_direction_vector * grad_dir;

    glm::vec3 dL_d_pos = dL_ddirection_vector;

    dL_dmeans[idx] = dL_d_pos;

    int offset = 0;

    for (int i = 0; i < INPUT_DIM_EMD * HIDDEN_DIM_1; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dweights_1[i];
    }
    offset += INPUT_DIM_EMD * HIDDEN_DIM_1;

    for (int i = 0; i < HIDDEN_DIM_1; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dbias_1[i];
    }
    offset += HIDDEN_DIM_1;

    for (int i = 0; i < HIDDEN_DIM_1 * HIDDEN_DIM_2; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dweights_2[i];
    }
    offset += HIDDEN_DIM_1 * HIDDEN_DIM_2;

    for (int i = 0; i < HIDDEN_DIM_2; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dbias_2[i];
    }
    offset += HIDDEN_DIM_2;

    for (int i = 0; i < HIDDEN_DIM_2 * OUTPUT_DIM; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dweights_3[i];
    }
    offset += HIDDEN_DIM_2 * OUTPUT_DIM;

    for (int i = 0; i < OUTPUT_DIM; ++i) {
        dL_dparas_one_gaus[offset + i] = dL_dbias_3[i];
    }
}


__device__ void computeColorFromSH(int idx, 
								   int deg, 
								   int max_coeffs, 
								   const glm::vec3* means, 
								   glm::vec3 campos, 
								   const float* shs, 
								   const bool* clamped, 
								   const glm::vec3* dL_dcolor, 
								   glm::vec3* dL_dmeans, 
								   glm::vec3* dL_dshs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0) {

		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1) {

			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2) {

				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });


	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) renderCUDA(const float* __restrict__ dL_dout_color,
																const float* __restrict__ means_3d,
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
	__shared__ float2    collected_signal[BLOCK_SIZE];


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

	float accum_rec_real = 0.0f;
	float accum_rec_imag = 0.0f;

	float last_sig_amp = 0.0f;
	float last_sig_pha = 0.0f;

	float dL_dpixel[C] = { 0 };
	if (inside) {
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dout_color[i * H * W + pix_id];
		}
	}

	const float d_loss_d_received_signal_real = dL_dpixel[0];
	const float d_loss_d_received_signal_imag = dL_dpixel[1];


	const float const_atten = 0.99f;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {

		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {

			const int gau_idx = bin_point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = gau_idx;

			collected_signal[block.thread_rank()] = make_float2(geom_rgb[C * gau_idx + 0], 
																geom_rgb[C * gau_idx + 1]);

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
			// Keep track of current position in range
			contributor--;
			if (contributor >= last_contributor) {
				continue;
			}

			int gau_idx = collected_id[j];
			glm::vec3 gau_pos = collected_means_3d[j];

			// here covMatrix is the inverse of the covariance matrix
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

			const float G = exp(power);
			const float temp_alpha = gau_atten * G;
			const float alpha = min(const_atten, temp_alpha);
			if (alpha < 1.0f / 255.0f) {
				continue;
			}

			T = T / (1.f - alpha);

			float dL_dsig_amp_real = cosf(sig_pha) * alpha * T * d_loss_d_received_signal_real;
			float dL_dsig_amp_imag = sinf(sig_pha) * alpha * T * d_loss_d_received_signal_imag;

			float d_loss_d_sig_amp = dL_dsig_amp_real + dL_dsig_amp_imag;

			float dL_dsig_pha_real = -sig_amp * sinf(sig_pha) * alpha * T * d_loss_d_received_signal_real;
			float dL_dsig_pha_imag =  sig_amp * cosf(sig_pha) * alpha * T * d_loss_d_received_signal_imag;

			float d_loss_d_sig_pha = dL_dsig_pha_real + dL_dsig_pha_imag;

			atomicAdd(&(dL_dcolors[2 * gau_idx + 0]), d_loss_d_sig_amp);
			atomicAdd(&(dL_dcolors[2 * gau_idx + 1]), d_loss_d_sig_pha);

			accum_rec_real = last_alpha * last_sig_amp * cosf(last_sig_pha) + (1.0 - last_alpha) * accum_rec_real;
			accum_rec_imag = last_alpha * last_sig_amp * sinf(last_sig_pha) + (1.0 - last_alpha) * accum_rec_imag;

			float dL_dalpha_real = (sig_amp * cosf(sig_pha) - accum_rec_real) * d_loss_d_received_signal_real;
			float dL_dalpha_imag = (sig_amp * sinf(sig_pha) - accum_rec_imag) * d_loss_d_received_signal_imag;

			float dL_dalpha = dL_dalpha_real + dL_dalpha_imag;
			dL_dalpha *= T;

			last_sig_amp = sig_amp;
			last_sig_pha = sig_pha;
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
									   *rx_pos, 
									   radius_rx, 
									   covMatrix, 
									   dL_dpower, 
									   dL_dgau_pos, 
									   dL_dinvCovMatrix);

			atomicAdd(&(grad_means_3d[3 * gau_idx + 0]), dL_dgau_pos.x);
			atomicAdd(&(grad_means_3d[3 * gau_idx + 1]), dL_dgau_pos.y);
			atomicAdd(&(grad_means_3d[3 * gau_idx + 2]), dL_dgau_pos.z);

			//  initializes all elements of the array dL_dcov3d to zero.
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
					  bool debug)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(dL_dout_color,
												  means_3d,
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



template<int C>
__global__ void preprocessCUDA(const int P,
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
							   bool debug)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
		
	computeColorFromSH(idx,
					   sh_degree_active,
					   sh_coef_len_max,
					   (glm::vec3*)means_3d,
					   *rx_pos,
					   signal_precomp,
					   geom_clamped,
					   (glm::vec3*)dL_dcolor,
					   (glm::vec3*)grad_means_3d,
					   (glm::vec3*)grad_signal_precomp);



}


void BACKWARD::preprocess(const int P,
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
						  bool debug)
{
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (P, 
																 means_3d, 
																 signal_precomp, 
																 rx_pos, 
																 radius_rx, 
																 tx_pos,
																 sh_degree_active, 
																 sh_coef_len_max, 
																 geom_clamped, 
																 dL_dcolor, 
																 grad_means_3d, 
																 grad_signal_precomp, 
																 debug);
}





