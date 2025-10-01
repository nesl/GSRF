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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <stdbool.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <math.h>

#include <glm/glm.hpp>          // Basic GLM functionalities
#include <cuComplex.h>


#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)


__device__ const float PI = 3.14159265358979323846;

__device__ const float light_speed = 3.0e8f;
__device__ const float signal_freq = 2.4e9f;


// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;

__device__ const float SH_C1 = 0.4886025119029199f;

__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};

__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};



__forceinline__ __device__ float dnormvdz(float3 v, 
                                          float3 dv
                                          )
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}


__forceinline__ __device__ float3 dnormvdv(float3 v, 
                                           float3 dv
                                           )
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}


__forceinline__ __device__ float4 dnormvdv(float4 v, 
                                           float4 dv
                                           )
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}


__forceinline__ __device__ float sin_func(float x) {
    return sinf(x);
}


__forceinline__ __device__ float cos_func(float x) {
    return cosf(x);
}


__forceinline__ __device__ float sin_derivative(float x) {
    return cosf(x);
}


__forceinline__ __device__ float cos_derivative(float x) {
    return -sinf(x);
}


__forceinline__ __device__ void create_embedding_fn(const float* input_data, float* output_data) {
    int out_idx = 0;

    // Include input
    for (int i = 0; i < INPUT_DIM_DIR; ++i) {
        output_data[out_idx + i] = input_data[i];
    }
    out_idx += INPUT_DIM_DIR;

    float max_freq = (float)MAX_FREQ_LOG2;
    int N_freqs = NUM_FREQS;

    for (int i = 0; i < N_freqs; ++i) {
        float freq = powf(2.0f, i * max_freq / (N_freqs - 1));

        for (int j = 0; j < INPUT_DIM_DIR; ++j) {
            output_data[out_idx + j] = sin_func(input_data[j] * freq);
        }
        out_idx += INPUT_DIM_DIR;

        for (int j = 0; j < INPUT_DIM_DIR; ++j) {
            output_data[out_idx + j] = cos_func(input_data[j] * freq);
        }
        out_idx += INPUT_DIM_DIR;
    }
}


__forceinline__ __device__ float absLeakyRelu(float x, 
                                              float alpha = 0.01
                                              ) 
{

    float leakyRelu = x >= 0 ? x : alpha * x;

    return fabsf(leakyRelu);

}


__forceinline__ __device__ float sigmoidTimesTwoPi(float x
                                                  ) 
{

    float sigmoidValue = 1.0f / (1.0f + expf(-x));
    
    return sigmoidValue * PI * 2;
}


__forceinline__ __device__ float computeGradient_dL_dGauSignalX(float dL_d_sa, 
                                                                float gau_signal_x, 
                                                                float alpha = 0.01
                                                                ) 
{
    float gradient = 0.0;

    // Determine the derivative of absLeakyRelu w.r.t. gau_signal.x
    if (gau_signal_x > 0) {

        gradient = dL_d_sa * 1.0; // Derivative is 1 for x > 0
        
    } else {
        gradient = dL_d_sa * alpha; // Derivative is alpha for x <= 0
    }

    return gradient;
}



__forceinline__ __device__ void cartesian_to_spherical(const glm::vec3& p_orig, 
                                                       const glm::vec3& rx_pos, 
                                                       const float* cov3d, 
                                                       float2& mean_2d
                                                       )
{
    glm::vec3 p_prime = p_orig - rx_pos;
    float x_prime = p_prime.x;
    float y_prime = p_prime.y;
    float z_prime = p_prime.z;

    float theta = atan2f(y_prime, x_prime);

    float theta_deg = glm::degrees(theta);
    if (theta_deg < 0.0f) {
        theta_deg = 360.0f + theta_deg;
    }

    float phi = atan2f(sqrtf(x_prime * x_prime + y_prime * y_prime), z_prime);


    float phi_deg = glm::degrees(phi);

    mean_2d = make_float2(theta_deg, phi_deg);

}


__forceinline__ __device__ void cartesian_to_lambert_azimuthal(const glm::vec3& p_orig, 
                                                               const glm::vec3& rx_pos, 
                                                               const float* cov3d, 
                                                               float2& mean_2d
                                                               )
{
    glm::vec3 p_prime = p_orig - rx_pos;
    float x_prime = p_prime.x;
    float y_prime = p_prime.y;
    float z_prime = p_prime.z;

    // Normalize coordinates
    float r = sqrtf(x_prime * x_prime + y_prime * y_prime + z_prime * z_prime);
    float x_prime_norm = x_prime / r;
    float y_prime_norm = y_prime / r;
    float z_prime_norm = z_prime / r;

    // Lambert Azimuthal Equal-Area Projection
    float k = sqrtf(2.0f / (1.0f + z_prime_norm));
    mean_2d = make_float2(k * x_prime_norm, k * y_prime_norm);

}


__forceinline__ __device__ void check_intersection(const glm::vec3& p_orig, 
												      const float* cov3d, 
												      const glm::vec3& rx_pos, 
												      const glm::vec3& tile_center,
                                                      int16_t& result
                                                      )
{

	glm::mat3 covMatrix = glm::mat3(cov3d[0], cov3d[1], cov3d[2], 
                                    cov3d[1], cov3d[3], cov3d[4], 
                                    cov3d[2], cov3d[4], cov3d[5]);

    glm::mat3 invCovMatrix = glm::inverse(covMatrix);

    glm::vec3 d = glm::normalize(tile_center);

	// Vector from line origin to ellipsoid center.
	glm::vec3 m_minus_rx_pos = rx_pos - p_orig;

	float k = 9.0f;

    float A = glm::dot(d, invCovMatrix * d);

    float B = 2.0f * glm::dot(d, invCovMatrix * m_minus_rx_pos);

    float C = glm::dot(m_minus_rx_pos, invCovMatrix * m_minus_rx_pos) - k;

    float discriminant = B * B - 4 * A * C;


	if (discriminant > 0.0f) {
		result = 1;
	}

}


__forceinline__ __device__ int maxHist(int16_t row[MAX_GRID_X], 
                                       int width, 
                                       int& start, 
                                       int& end
                                       )
{
    int stack[MAX_GRID_X], stackIdx = 0;
    int max_area = 0, area = 0;
    start = 0;
    end = 0;

    for (int i = 0; i <= width; i++) {

        int currHeight = (i == width) ? 0 : row[i];
        if (stackIdx == 0 || currHeight >= row[stack[stackIdx - 1]]) {

            stack[stackIdx++] = i;

        } else {

            while (stackIdx > 0 && currHeight < row[stack[stackIdx - 1]]) {
                int height = row[stack[--stackIdx]];
                int width = stackIdx == 0 ? i : i - stack[stackIdx - 1] - 1;
                area = height * width;
                if (area > max_area) {
                    max_area = area;
                    start = stackIdx > 0 ? stack[stackIdx - 1] + 1 : 0;
                    end = i - 1;
                }
            }

            stack[stackIdx++] = i;
        }
    }
    return max_area;
}


__forceinline__ __device__ void computeLargestRectangle(int16_t covered[MAX_GRID_Y][MAX_GRID_X], 
                                                        const dim3 tile_grid, 
                                                        uint2& rect_min, 
                                                        uint2& rect_max
                                                        )
{
    
    int start_t, end_t;
    int result = maxHist(covered[0], tile_grid.x, start_t, end_t);

    int max_area = result;

    int finalStartCol = 0, finalEndCol = 0, finalStartRow = 0, finalEndRow = 0;

    if (max_area != 0) {
        finalEndRow = 1;
    }

    // Process each row
    for (int i = 1; i < tile_grid.y; i++) {

        for (int j = 0; j < tile_grid.x; j++) {

            // Update the current row if there's a "true" value directly above
            if (covered[i][j]) {

                covered[i][j] += covered[i - 1][j];

            }
        }

        int startCol, endCol;

        int current_area = maxHist(covered[i], tile_grid.x, startCol, endCol);

        if (current_area > max_area) {

            max_area = current_area;

            finalStartCol = startCol;
            finalEndCol = endCol;

            // Backtrack to find the actual starting row of the rectangle
            finalStartRow = i;

            for (int up = i; up >= 0; --up) {
                
                // Flag to check if all elements are positive
                bool allPositive = true; 

                for (int k = startCol; k <= endCol; ++k) {

                    if (covered[up][k] <= 0) {

                        allPositive = false;
                        break;
                    }
                }
                
                if (allPositive) {

                    // Updates finalStartRow if all elements are positive
                    finalStartRow = up;
                } else {

                    // Breaks the outer loop if not all elements are positive
                    break;
                }
            }
            finalEndRow = i;

        }
    }

    rect_min.x = finalStartCol;
    rect_min.y = finalStartRow;
    rect_max.x = finalEndCol;
    rect_max.y = finalEndRow;
}


__forceinline__ __device__ void getRect(const float2 p, 
                                        int max_radius, 
                                        uint2& rect_min, 
                                        uint2& rect_max, 
                                        dim3 grid
                                        )
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
    
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}


__forceinline__ __device__ void getRect_v2(const glm::vec3 p_orig,
                                           const float* cov3d,
                                           const float* spectrum_3d,
                                           const glm::vec3 rx_pos,
                                           const dim3 tile_grid,
                                           uint2& rect_min,
                                           uint2& rect_max
                                           )
{

	int16_t covered[MAX_GRID_Y][MAX_GRID_X]{}; // Initialize to 0


    // Iterate over all tiles to find influenced ones, tile_grid: (23, 6, 1)
	// tile_grid.y: Represents the height of the grid, hence the number of rows.
	// j is the row index
	for (int j = 0; j < tile_grid.y; ++j) {

		// tile_grid.x: Represents the width of the grid, hence the number of columns.
		// i is the column index
		for (int i = 0; i < tile_grid.x; ++i) {
            
            int idx = j * tile_grid.x + i;

            glm::vec3 tile_center = glm::vec3( spectrum_3d[3 * idx + 0], 
											   spectrum_3d[3 * idx + 1], 
											   spectrum_3d[3 * idx + 2] );

            check_intersection(p_orig, cov3d, rx_pos, tile_center, covered[j][i]);

    	}
	}

	computeLargestRectangle(covered, tile_grid, rect_min, rect_max);

}


__forceinline__ __device__ void calculate_central_angle(float my_radius, 
                                                        float radius_rx, 
                                                        float& angle
                                                        ) 
{
    float theta_radians = 2 * asinf(my_radius / (2 * radius_rx));
    
    angle = theta_radians * (180.0f / PI);
}


__forceinline__ __device__ void getRect_v3(const float2 point_image, 
                                           float max_radius, 
                                           uint2& rect_min, 
                                           uint2& rect_max, 
                                           dim3 grid
                                           )
{
    // Calculate rectangle corners
    float rect_min_x = point_image.x - max_radius;
    float rect_min_y = point_image.y - max_radius;
    float rect_max_x = point_image.x + max_radius;
    float rect_max_y = point_image.y + max_radius;

    // Convert to grid coordinates
    rect_min.x = static_cast<unsigned int>(floorf(rect_min_x / BLOCK_X));
    rect_min.y = static_cast<unsigned int>(floorf(rect_min_y / BLOCK_Y));
    
    rect_max.x = static_cast<unsigned int>(ceilf(rect_max_x / BLOCK_X));
    rect_max.y = static_cast<unsigned int>(ceilf(rect_max_y / BLOCK_Y));

    rect_min.x = max(0, min(rect_min.x, grid.x - 1));
    rect_min.y = max(0, min(rect_min.y, grid.y - 1));
    rect_max.x = max(0, min(rect_max.x, grid.x - 1));
    rect_max.y = max(0, min(rect_max.y, grid.y - 1));
    
}


__forceinline__ __device__ float calculatePerpendicularDistance(const glm::vec3& rx_pos, 
                                                                const glm::vec3& pix_dir, 
                                                                const glm::vec3& gau_pos
                                                                )
{
	glm::vec3 PA = rx_pos - gau_pos;

    glm::vec3 projection = (glm::dot(PA, pix_dir) / glm::dot(pix_dir, pix_dir)) * pix_dir;

	glm::vec3 perpendicularVector = PA - projection; 

	return glm::length(perpendicularVector); 

}


__forceinline__ __device__ float calculatePathLengthThroughGaussian(const glm::vec3& p_orig, 
                                                                    const glm::mat3& invCovMatrix,
                                                                    const glm::vec3& rx_pos, 
                                                                    const glm::vec3& d
                                                                    )
{

	glm::vec3 m_minus_rx_pos = rx_pos - p_orig;

	float k = 9.0f;

    float A = glm::dot(d, invCovMatrix * d);

    float B = 2.0f * glm::dot(d, invCovMatrix * m_minus_rx_pos);

    float C = glm::dot(m_minus_rx_pos, invCovMatrix * m_minus_rx_pos) - k;

    float discriminant = B * B - 4 * A * C;

    if (discriminant <= 0) {
        return 0.0f;
    }

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-B + sqrtDisc) / (2 * A);
    float t2 = (-B - sqrtDisc) / (2 * A);

    // Calculate intersection points
    glm::vec3 intersectionPoint1 = rx_pos + t1 * d;
    glm::vec3 intersectionPoint2 = rx_pos + t2 * d;

    float pathLength = glm::distance(intersectionPoint1, intersectionPoint2);

    return pathLength;
}


__forceinline__ __device__ glm::vec3 findClosestPoint(const glm::vec3& rx_pos, 
                                                      const glm::vec3& pix_dir, 
                                                      const glm::vec3& gau_pos
                                                      )
{
    glm::vec3 rx_to_gau = gau_pos - rx_pos;

    float t = glm::dot(rx_to_gau, pix_dir) / glm::dot(pix_dir, pix_dir);

    return rx_pos + t * pix_dir;
}


__forceinline__ __device__ glm::vec2 computeColorFromSH_forward(glm::vec3 dir_t, 
                                                        int deg, 
                                                        glm::vec2* sh
                                                        )
{

	glm::vec3 dir = glm::vec3(-dir_t.x, -dir_t.y, -dir_t.z);

	dir = dir / glm::length(dir);

	glm::vec2 result = SH_C0 * sh[0];

	if (deg > 0) {

		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		result = result 
					- SH_C1 * y * sh[1]
					+ SH_C1 * z * sh[2]
					- SH_C1 * x * sh[3];

		if (deg > 1) {

			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			result = result 
						+ SH_C2[0] * xy                    * sh[4]
						+ SH_C2[1] * yz                    * sh[5]
						+ SH_C2[2] * (2.0f * zz - xx - yy) * sh[6]
						+ SH_C2[3] * xz                    * sh[7] 
						+ SH_C2[4] * (xx - yy)             * sh[8];

			if (deg > 2) {

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

    return result;


}


__forceinline__ __device__ glm::vec2 computeColorFromSH_forward_v2(glm::vec3 pix_pos, 
                                                                   glm::vec3 gau_pos,
                                                                   int deg, 
                                                                   const glm::vec2* sh
                                                                   )
{

    glm::vec3 dir = gau_pos - pix_pos;
	dir = dir / glm::length(dir);

	glm::vec2 result = SH_C0 * sh[0];

	if (deg > 0) {

		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		result = result 
					- SH_C1 * y * sh[1]
					+ SH_C1 * z * sh[2]
					- SH_C1 * x * sh[3];

		if (deg > 1) {

			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			result = result 
						+ SH_C2[0] * xy                    * sh[4]
						+ SH_C2[1] * yz                    * sh[5]
						+ SH_C2[2] * (2.0f * zz - xx - yy) * sh[6]
						+ SH_C2[3] * xz                    * sh[7] 
						+ SH_C2[4] * (xx - yy)             * sh[8];

			if (deg > 2) {

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

    return result;


}


__forceinline__ __device__ glm::vec2 computeColorFromSH_v2(glm::vec3 dir_t, 
                                                          int deg, 
                                                          glm::vec2* sh
                                                          )
{

	glm::vec3 dir = glm::vec3(-dir_t.x, -dir_t.y, -dir_t.z);

	dir = dir / glm::length(dir);

	glm::vec2 result = SH_C0 * sh[0];

	if (deg > 0) {

		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		result = result 
					- SH_C1 * y * sh[1]
					+ SH_C1 * z * sh[2]
					- SH_C1 * x * sh[3];

		if (deg > 1) {

			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			result = result 
						+ SH_C2[0] * xy                    * sh[4]
						+ SH_C2[1] * yz                    * sh[5]
						+ SH_C2[2] * (2.0f * zz - xx - yy) * sh[6]
						+ SH_C2[3] * xz                    * sh[7] 
						+ SH_C2[4] * (xx - yy)             * sh[8];

			if (deg > 2) {

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

	return glm::max(result, 0.0f);

}


__forceinline__ __device__ glm::ivec4 affected_tiles_v2(float azimuth_center, 
                                                        float elevation_center,
                                                        float sphere_radius, 
                                                        float hemisphere_radius=0.5, 
                                                        int tile_size=16
                                                        )
{

    float angular_radius;
    if (sphere_radius >= hemisphere_radius) {

        angular_radius = 90.0f; 

    } else {
        angular_radius = glm::degrees(asinf(sphere_radius / hemisphere_radius));
    }

    float min_azimuth = fmaxf(0.0f, azimuth_center - angular_radius);
    float max_azimuth = fminf(360.0f, azimuth_center + angular_radius);

    float min_elevation = fmaxf(0.0f, elevation_center - angular_radius);
    float max_elevation = fminf(90.0f, elevation_center + angular_radius);

    // Convert these ranges into tile indices
    int azimuth_tile_start = static_cast<int>(min_azimuth / tile_size);
    int azimuth_tile_end = static_cast<int>(max_azimuth / tile_size);

    int elevation_tile_start = static_cast<int>(min_elevation / tile_size);
    int elevation_tile_end = static_cast<int>(max_elevation / tile_size);

    return glm::ivec4(elevation_tile_start, elevation_tile_end, azimuth_tile_start, azimuth_tile_end);
}


__forceinline__ __device__ float2 find_intersection_idx(const glm::vec3& rx_pos, 
                                                        const glm::vec3& p_orig, 
                                                        const float radius, 
                                                        const int grid_size, 
                                                        const int idx
                                                        )
{
    glm::vec3 direction_vector = p_orig - rx_pos;
    glm::vec3 normalized_vector = glm::normalize(direction_vector);
    glm::vec3 intersection_point = rx_pos + radius * normalized_vector;

    float azimuth_center = glm::degrees(glm::atan(intersection_point.y - rx_pos.y, intersection_point.x - rx_pos.x));

    if (azimuth_center < 0) {
        azimuth_center += 360.0f; 
    }

    float relative_z = intersection_point.z - rx_pos.z;
    float elevation_center = glm::degrees(glm::asin(relative_z / radius));

    elevation_center = glm::abs(elevation_center);

    return make_float2(azimuth_center, elevation_center);

}


__forceinline__ __device__ float calculate_exponent(const glm::vec3& pix_pos, 
                                                    const glm::vec3& gau_pos, 
                                                    const glm::vec3& rx_pos, 
                                                    const float radius, 
                                                    const glm::mat3& invCovMatrix
                                                    )
{
    glm::vec3 direction_vector = gau_pos - rx_pos;

    float dir_length = glm::length(direction_vector);

    glm::vec3 normalized_vector = direction_vector / dir_length;

    glm::vec3 intersection_point = rx_pos + radius * normalized_vector;

    glm::vec3 diff = pix_pos - intersection_point;

    glm::vec3 tmp = invCovMatrix * diff;

    float power = -0.5f * glm::dot(diff, tmp);

	return power;

}


// This function calculates gau_signal_amp * exp(1j * gau_signal_pha)
// Using cuComplex library functions for complex numbers
__forceinline__ __device__ cuComplex complexExpMult(float gau_signal_amp, 
                                                    float gau_signal_pha
                                                    ) 
{
    cuComplex expPhase = make_cuFloatComplex(cosf(gau_signal_pha), sinf(gau_signal_pha));
    
    return cuCmulf(expPhase, make_cuFloatComplex(gau_signal_amp, 0.0));
}


__forceinline__ __device__ cuComplex updateReceiveSignal(float signal_amp, 
                                                         float signal_pha, 
                                                         float atten_amp_cum, 
                                                         float atten_amp_pha, 
                                                         float path_loss
                                                         )
{
    cuComplex complexSignal = complexExpMult(signal_amp, signal_pha);
    
    cuComplex complexCur = complexExpMult(atten_amp_cum, atten_amp_pha);
    
    cuComplex product1 = cuCmulf(complexSignal, complexCur);

    cuComplex pathLossProduct = cuCmulf(make_cuComplex(path_loss, 0), product1);
    
	return pathLossProduct;

}


__forceinline__ __device__ cuComplex updateReceiveSignal_v2(float signal_amp, 
                                                            float signal_pha, 
                                                            float atten_cur_amp, 
                                                            float atten_cur_pha, 
                                                            float atten_cum_amp, 
                                                            float atten_cum_pha, 
                                                            float path_loss
                                                            )
{
    cuComplex complex_signal = complexExpMult(signal_amp, signal_pha);
    
    cuComplex atten_cur = complexExpMult(atten_cur_amp, atten_cur_pha);

    cuComplex atten_cum = complexExpMult(atten_cum_amp, atten_cum_pha);
    
    cuComplex product = cuCmulf(complex_signal, atten_cur);

    product = cuCmulf(product, atten_cum);

    cuComplex pathLossProduct = cuCmulf(make_cuFloatComplex(path_loss, 0.0f), product);
    
	return pathLossProduct;

}


__forceinline__ __device__ cuComplex updateReceiveSignal_v3(float signal_amp, 
                                                            float signal_pha, 
                                                            float atten_cur_amp, 
                                                            float atten_cur_pha, 
                                                            float atten_cum_amp, 
                                                            float atten_cum_pha, 
                                                            float path_loss_amp,
                                                            float path_loss_pha
                                                            )
{
    cuComplex complex_signal = complexExpMult(signal_amp, signal_pha);
    
    cuComplex atten_cur = complexExpMult(atten_cur_amp, atten_cur_pha);

    cuComplex atten_cum = complexExpMult(atten_cum_amp, atten_cum_pha);

    cuComplex path_loss = complexExpMult(path_loss_amp, path_loss_pha);

    cuComplex product = cuCmulf(complex_signal, atten_cur);

    product = cuCmulf(product, atten_cum);
    
	return cuCmulf(product, path_loss);

}


__forceinline__ __device__ cuComplex updateReceiveSignal_v4(float signal_amp, 
                                                            float signal_pha, 
                                                            float atten_cum_amp, 
                                                            float atten_cum_pha, 
                                                            float path_loss_amp,
                                                            float path_loss_pha
                                                            )
{
    cuComplex complex_signal = complexExpMult(signal_amp, signal_pha);
    
    cuComplex atten_cum = complexExpMult(atten_cum_amp, atten_cum_pha);

    cuComplex path_loss = complexExpMult(path_loss_amp, path_loss_pha);

    cuComplex product = cuCmulf(complex_signal, atten_cum);

	return cuCmulf(product, path_loss);

}


__forceinline__ __device__ cuComplex updateReceiveSignal_bak(float path_loss, 
                                                             float signal_amp, 
                                                             float signal_pha, 
                                                             float test_T_amp, 
                                                             float test_T_pha
                                                             ) 
{
    cuComplex complexSignal = complexExpMult(signal_amp, signal_pha);
    
    cuComplex complexTestT = complexExpMult(test_T_amp, test_T_pha);
    
    cuComplex product = cuCmulf(complexSignal, complexTestT);
    
    cuComplex pathLossProduct = cuCmulf(make_cuComplex(path_loss, 0), product);
    
	return pathLossProduct;

}


__forceinline__ __device__ float generate_float01(unsigned int seed) {
    const unsigned int a = 1664525;
    const unsigned int c = 1013904223;
    
    unsigned int nextSeed = a * seed + c;
    
    return static_cast<float>(nextSeed) / static_cast<float>(UINT_MAX);
}


__forceinline__ __device__ cuFloatComplex gradientCuCabsf(cuFloatComplex y, 
                                                          float dL_dx
                                                          ) 
{

    float a = cuCrealf(y);
    float b = cuCimagf(y);
    float magnitude = cuCabsf(y);

    float grad_a = 0.0f;
    float grad_b = 0.0f;

    if (magnitude > 0.0f) {
        grad_a = a / magnitude;
        grad_b = b / magnitude;
    }

    grad_a *= dL_dx;
    grad_b *= dL_dx;

    return make_cuFloatComplex(grad_a, grad_b);
}


__forceinline__ __device__ void computeGradient_dL_dsa(const float signal_amp, 
                                                       const float signal_pha, 
                                                       const float atten_cur_amp, 
                                                       const float atten_cur_pha, 
                                                       const float atten_cum_amp, 
                                                       const float atten_cum_pha, 
                                                       const float path_loss,
                                                       const float dL_dr, 
                                                       const float dL_dtheta,
                                                       cuComplex& dL_dsa
                                                       )
{
    cuComplex dL_d_s_a_component = complexExpMult(atten_cur_amp, atten_cur_pha);
    
    dL_d_s_a_component = cuCmulf(dL_d_s_a_component, complexExpMult(atten_cum_amp, atten_cum_pha));
    dL_d_s_a_component = cuCmulf(dL_d_s_a_component, make_cuFloatComplex(path_loss, 0));

    cuComplex dtemp_signal_d_s_a = make_cuFloatComplex(cos(signal_pha), sin(signal_pha));

    dtemp_signal_d_s_a = cuCmulf(dtemp_signal_d_s_a, dL_d_s_a_component);

    cuComplex grad_temp_signal = make_cuFloatComplex(dL_dr, dL_dtheta);

    dL_dsa = cuCmulf(grad_temp_signal, dtemp_signal_d_s_a);

}


__forceinline__ __device__ void computeGradient_dL_dsp(const float signal_amp, 
                                                       const float signal_pha, 
                                                       const float atten_cur_amp, 
                                                       const float atten_cur_pha, 
                                                       const float atten_cum_amp, 
                                                       const float atten_cum_pha, 
                                                       const float path_loss,
                                                       const float dL_dr, 
                                                       const float dL_dtheta,
                                                       cuComplex& dL_dsp
                                                       )
{
    cuComplex dL_d_s_a_component = complexExpMult(atten_cur_amp, atten_cur_pha);
    
    dL_d_s_a_component = cuCmulf(dL_d_s_a_component, complexExpMult(atten_cum_amp, atten_cum_pha));
    dL_d_s_a_component = cuCmulf(dL_d_s_a_component, make_cuComplex(path_loss, 0));

    
    cuComplex dtemp_signal_d_s_p = make_cuFloatComplex(signal_amp * sin(signal_pha) * -1.0f,  // Real
                                                       signal_amp * cos(signal_pha));         // Imaginary

    dtemp_signal_d_s_p = cuCmulf(dtemp_signal_d_s_p, dL_d_s_a_component);

    cuComplex grad_temp_signal = make_cuFloatComplex(dL_dr, dL_dtheta);

    dL_dsp = cuCmulf(grad_temp_signal, dtemp_signal_d_s_p);

}


__forceinline__ __device__ float dAbsLeakyRelu_dx(float x, 
                                                  float alpha = 0.01
                                                  )
{   

    return x >= 0 ? 1.0f : -alpha;

}


__forceinline__ __device__ float dSigmoidTimesTwoPi_dy(float y) 
{
    float sigmoid_val = 1.0f / (1.0f + expf(-y));

    return sigmoid_val * (1.0f - sigmoid_val) * (PI * 2);

}


__forceinline__ __device__ void gradcomputeColorFromSH_backward(glm::vec3 dir_t, 
									   int deg, 
									   glm::vec2* sh,
									   const float dL_dgau_signal_x,	
									   const float dL_dgau_signal_y,
									   glm::vec2* dL_dsh
									   )
{
	glm::vec3 dir = glm::vec3(-dir_t.x, -dir_t.y, -dir_t.z);

	dir = dir / glm::length(dir);

	glm::vec2 dL_dRGB = glm::vec2(dL_dgau_signal_x, dL_dgau_signal_y);

	glm::vec2 dRGBdx(0, 0);
	glm::vec2 dRGBdy(0, 0);
	glm::vec2 dRGBdz(0, 0);

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

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


}



__forceinline__ __device__ void gradcomputeColorFromSH_backward_v2(glm::vec3 pix_pos, 
                                                                   glm::vec3 gau_pos,
                                                                   int deg, 
                                                                   const glm::vec2* sh, 
                                                                   const float dL_dgau_signal_x, 
                                                                   const float dL_dgau_signal_y,
                                                                   glm::vec3& dL_dmean, 
                                                                   glm::vec2* dL_dsh)
{
    glm::vec3 dir_orig = gau_pos - pix_pos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

    glm::vec2 dL_dRGB = glm::vec2(dL_dgau_signal_x, dL_dgau_signal_y);

    glm::vec2 dRGBdx(0, 0);
	glm::vec2 dRGBdy(0, 0);
	glm::vec2 dRGBdz(0, 0);

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0) {

		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 =  SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;

		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz =  SH_C1 * sh[2];

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

	float3 dL_dmean_nrom = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

    dL_dmean = glm::vec3(dL_dmean_nrom.x, dL_dmean_nrom.y, dL_dmean_nrom.z);

}



__forceinline__ __device__ void calculateGradientOfLossWrtGauPos_dist(const glm::vec3& p_orig, 
                                                                      const glm::mat3& invCovMatrix,
                                                                      const glm::vec3& rx_pos, 
                                                                      const glm::vec3& d, 
                                                                      float dL_dPathLength, 
                                                                      glm::vec3&dL_dp_orig, 
                                                                      glm::mat3&dL_dInvCovMatrix
                                                                      )
{
    // Vector from line origin to ellipsoid center.
	glm::vec3 m_minus_rx_pos = rx_pos - p_orig;

	float k = 9.0f;

    float A = glm::dot(d, invCovMatrix * d);

    float B = 2.0f * glm::dot(d, invCovMatrix * m_minus_rx_pos);

    float C = glm::dot(m_minus_rx_pos, invCovMatrix * m_minus_rx_pos) - k;

    float discriminant = B * B - 4 * A * C;

    if (discriminant <= 0) {
        // return 0.0f;
        dL_dInvCovMatrix = glm::mat3(0.0f);
    }

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-B + sqrtDisc) / (2 * A);
    float t2 = (-B - sqrtDisc) / (2 * A);

    // Calculate intersection points
    glm::vec3 intersectionPoint1 = rx_pos + t1 * d;
    glm::vec3 intersectionPoint2 = rx_pos + t2 * d;

    float pathLength = glm::distance(intersectionPoint1, intersectionPoint2);

    glm::vec3 dL_dp1 = ((intersectionPoint1 - intersectionPoint2) / pathLength) * dL_dPathLength;

    glm::vec3 dL_dp2 = -dL_dp1;

    glm::vec3 dL_dt1_vec = dL_dp1 * d;
    glm::vec3 dL_dt2_vec = dL_dp2 * d;

    float dL_dt1 = glm::dot(dL_dt1_vec, d);
    float dL_dt2 = glm::dot(dL_dt2_vec, d);

    // Derivative calculation for A
    float dt1_dA = (B / (2 * A * A * sqrtDisc)) - (1 / (2 * A)) + (sqrtDisc / (2 * A * A));
    float dt2_dA = -(B / (2 * A * A * sqrtDisc)) - (1 / (2 * A)) - (sqrtDisc / (2 * A * A));

    float dL_dA = dL_dt1 * dt1_dA + dL_dt2 * dt2_dA;

    // Derivative calculation for B
    float dt1_dB = (-1 / (2 * A)) + (B / (2 * A * sqrtDisc));
    float dt2_dB = (-1 / (2 * A)) - (B / (2 * A * sqrtDisc));

    float dL_dB = dL_dt1 * dt1_dB + dL_dt2 * dt2_dB;

    // Derivative calculation for C
    float dt1_dC = -1 / sqrtDisc;
    float dt2_dC = 1 / sqrtDisc;

    float dL_dC = dL_dt1 * dt1_dC + dL_dt2 * dt2_dC;

    // Gradient of L with respect to B
    glm::vec3 dL_dB_vec = glm::vec3(dL_dB * 2.0f * invCovMatrix * d);

    // Gradient of L with respect to C
    glm::vec3 dL_dC_vec = glm::vec3(dL_dC * 2.0f * invCovMatrix * (rx_pos - p_orig)); 

    // Negation due to derivative of m_minus_rx_pos w.r.t. p_orig
    // glm::vec3 dL_dp_orig = -(dL_dB_vec + dL_dC_vec); 
    dL_dp_orig = -(dL_dB_vec + dL_dC_vec); 


    glm::mat3 dA_dInvCovMatrix = outerProduct(d, d); // glm::outerProduct might not exist; conceptual
    glm::mat3 dB_dInvCovMatrix = 2.0f * outerProduct(d, m_minus_rx_pos);
    glm::mat3 dC_dInvCovMatrix = outerProduct(m_minus_rx_pos, m_minus_rx_pos);


    dL_dInvCovMatrix = dL_dA * dA_dInvCovMatrix + 
                       dL_dB * dB_dInvCovMatrix + 
                       dL_dC * dC_dInvCovMatrix;

}


__forceinline__ __device__ void calculate_exponent_grad(const glm::vec3& rx_pos, 
                                                        const glm::vec3& pix_dir, 
                                                        const glm::vec3& gau_pos, 
                                                        const glm::mat3& invCovMatrix, 
                                                        float dL_dpower, 
                                                        glm::vec3& dL_dgau_pos,
                                                        glm::mat3& dL_dinvCovMatrix
                                                        )
{

    glm::vec3 rx_to_gau = gau_pos - rx_pos;
    float t = glm::dot(rx_to_gau, pix_dir) / glm::dot(pix_dir, pix_dir);
    glm::vec3 closestPoint =  rx_pos + t * pix_dir;

    // the difference vector
    glm::vec3 diff = closestPoint - gau_pos;

	// the scalar for the exponential function
    float power = -0.5f * glm::dot(diff, invCovMatrix * diff);

    glm::vec3 dL_ddiff = -dL_dpower * (invCovMatrix * diff);

    glm::vec3 dL_dclosestPoint = dL_ddiff;
    dL_dgau_pos = -1.0f * dL_ddiff;

    dL_dgau_pos = dL_dgau_pos + dL_dclosestPoint * (pix_dir / glm::dot(pix_dir, pix_dir));

    glm::mat3 diff_outer_diff = glm::outerProduct(diff, diff);
    
    dL_dinvCovMatrix = dL_dpower * (-0.5f) * diff_outer_diff;


}


__forceinline__ __device__ void calculate_exponent_grad_v2(const glm::vec3& gau_pos, 
                                                           const glm::vec3& pix_pos, 
                                                           const float radius, 
                                                           const glm::mat3& invCovMatrix,
                                                           float dL_d_power, 
                                                           glm::vec3& dL_d_gau_pos,
                                                           glm::mat3& dL_d_invCovMatrix
                                                           )
{
    glm::vec3 direction_vector = gau_pos - pix_pos;
    float distance = glm::length(direction_vector);
    glm::vec3 normalized_vector = direction_vector / distance;
    glm::vec3 intersection_point = pix_pos + radius * normalized_vector;
    glm::vec3 diff = pix_pos - intersection_point;
    glm::vec3 tmp = invCovMatrix * diff;
    float power = -0.5f * glm::dot(diff, tmp);

    glm::mat3 I = glm::mat3(1.0f);
    glm::mat3 outer_product = glm::outerProduct(direction_vector, direction_vector);
    glm::mat3 term = I - (outer_product / (distance * distance));
    glm::mat3 grad_intersection_wrt_gau_pos = (radius / distance) * term;
    glm::vec3 grad_diff_wrt_gau_pos = -grad_intersection_wrt_gau_pos * direction_vector; 

    glm::vec3 grad_power_wrt_diff = -0.5f * invCovMatrix * diff;  

    dL_d_gau_pos = dL_d_power * glm::dot(grad_power_wrt_diff, grad_diff_wrt_gau_pos) * grad_diff_wrt_gau_pos;


    dL_d_invCovMatrix = dL_d_power * (-0.5f * glm::outerProduct(diff, diff));
}


__forceinline__ __device__ void calculate_exponent_grad_v3(const glm::vec3& pix_pos, 
                                                           const glm::vec3& gau_pos, 
                                                           const glm::vec3& rx_pos, 
                                                           const float radius, 
                                                           const glm::mat3& invCovMatrix,
                                                           float dL_d_power, 
                                                           glm::vec3& dL_d_gau_pos,
                                                           glm::mat3& dL_d_invCovMatrix
                                                           )
{
    // Compute the direction vector from rx_pos to gau_pos
    glm::vec3 direction_vector = gau_pos - rx_pos;

    float dir_length = glm::length(direction_vector);

    glm::vec3 normalized_vector = direction_vector / dir_length;

    glm::vec3 intersection_point = rx_pos + radius * normalized_vector;

    glm::vec3 diff = pix_pos - intersection_point;

    glm::vec3 tmp = invCovMatrix * diff;

    glm::vec3 d_power_d_diff = -1.0f * invCovMatrix * diff;

    glm::mat3 d_diff_d_intersection_point = -1.0f * glm::mat3(1.0f);

    glm::mat3 d_intersection_point_d_norm_dir = radius * glm::mat3(1.0f);

    glm::mat3 identity_matrix = glm::mat3(1.0f);
    glm::mat3 outer_norm = glm::outerProduct(direction_vector, direction_vector);
    glm::mat3 d_norm_dir_d_direction_vector = (1.0f / dir_length) * (identity_matrix - outer_norm / (dir_length * dir_length));

    glm::mat3 d_direction_vector_d_gau_pos = glm::mat3(1.0f);

    dL_d_gau_pos = dL_d_power * (d_power_d_diff * d_diff_d_intersection_point * d_intersection_point_d_norm_dir * d_norm_dir_d_direction_vector * d_direction_vector_d_gau_pos);

    dL_d_invCovMatrix = -0.5f * dL_d_power * glm::outerProduct(diff, diff);
}


__forceinline__ __device__ void calculateGradientWrtCov3d(const glm::mat3& dL_dinvCovMatrix, 
                                                          const glm::mat3& invCovMatrix, 
                                                          float* dL_dcov3d
                                                          )
{
    glm::mat3 dL_d_Cov = -invCovMatrix * dL_dinvCovMatrix * invCovMatrix;

    dL_dcov3d[0] = dL_d_Cov[0][0]; // A[0][0]
    dL_dcov3d[1] = dL_d_Cov[0][1]; // A[0][1] or A[1][0]
    dL_dcov3d[2] = dL_d_Cov[0][2]; // A[0][2] or A[2][0]
    dL_dcov3d[3] = dL_d_Cov[1][1]; // A[1][1]
    dL_dcov3d[4] = dL_d_Cov[1][2]; // A[1][2] or A[2][1]
    dL_dcov3d[5] = dL_d_Cov[2][2]; // A[2][2]
}


__forceinline__ __device__ float3 transformPoint4x3(const float3& p, 
                                                    const float* matrix
                                                    )
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}


__forceinline__ __device__ float4 transformPoint4x4(const float3& p, 
                                                    const float* matrix
                                                    )
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}


__forceinline__ __device__ float3 transformVec4x3(const float3& p, 
                                                  const float* matrix
                                                  )
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}


__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, 
                                                           const float* matrix
                                                           )
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}



__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}


__forceinline__ __device__ float relu(float x) {
    return x > 0 ? x : 0;
    
}


__forceinline__ __device__ float leaky_relu(float x, 
                                              float alpha = 0.01
                                              ) 
{

    return x >= 0 ? x : alpha * x;
}


__forceinline__ __device__ float sigmoid_derivative(float y) {
    // y is already the output of the sigmoid function
    return y * (1.0f - y);
}


__forceinline__ __device__ float leaky_relu_derivative(float x, float alpha = 0.01f) {
    return x >= 0 ? 1.0f : alpha;
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif



