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


__forceinline__ __device__ void cartesian_to_spherical(const glm::vec3& p_orig,
                                                       const glm::vec3& sphere_center,
                                                       const float* cov3d,
                                                       float2& mean_2d
                                                       )
{
    glm::vec3 p_prime = p_orig - sphere_center;
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

    // Convert from "angle from z-axis" to "elevation from horizontal"
    // to match the pixel grid where row 0 = elevation 1° (equator) and row 89 = elevation 90° (pole)
    float elev_deg = 90.0f - phi_deg;

    mean_2d = make_float2(theta_deg, elev_deg);

}


__forceinline__ __device__ void calculate_central_angle(float my_radius,
                                                        float sphere_radius,
                                                        float& angle
                                                        )
{
    float ratio = fminf(my_radius / (2 * sphere_radius), 1.0f);
    float theta_radians = 2 * asinf(ratio);

    angle = theta_radians * (180.0f / PI);
}


__forceinline__ __device__ void getRect_v3(const float2 point_image,
                                           float max_radius,
                                           uint2& rect_min,
                                           uint2& rect_max,
                                           dim3 grid
                                           )
{
    // point_image.y is elevation from horizontal: 0° at equator, 90° at pole.
    // Near the pole, azimuth lines converge, so a Gaussian's azimuth extent
    // must be expanded by 1/cos(elevation) to cover the correct tiles.
    float elev_rad = point_image.y * PI / 180.0f;
    float cos_elev = fmaxf(cosf(elev_rad), 0.02f);  // clamp to avoid division by zero
    float azimuth_radius = max_radius / cos_elev;

    // Calculate rectangle corners
    float rect_min_x = point_image.x - azimuth_radius;
    float rect_min_y = point_image.y - max_radius;
    float rect_max_x = point_image.x + azimuth_radius;
    float rect_max_y = point_image.y + max_radius;

    // Handle azimuth wrap-around: if bounding box crosses 0/360 boundary,
    // cover all azimuth tiles so the Gaussian contributes on both sides of the seam
    bool azimuth_wraps = (rect_min_x < 0.0f) || (rect_max_x >= (float)(grid.x * BLOCK_X));

    if (azimuth_wraps) {
        rect_min.x = 0;
        rect_max.x = grid.x - 1;
    } else {
        rect_min.x = static_cast<unsigned int>(floorf(rect_min_x / BLOCK_X));
        rect_max.x = static_cast<unsigned int>(ceilf(rect_max_x / BLOCK_X));
        rect_min.x = max(0, min(rect_min.x, grid.x - 1));
        rect_max.x = max(0, min(rect_max.x, grid.x - 1));
    }

    // Convert to grid coordinates (elevation doesn't wrap)
    rect_min.y = static_cast<unsigned int>(floorf(rect_min_y / BLOCK_Y));
    rect_max.y = static_cast<unsigned int>(ceilf(rect_max_y / BLOCK_Y));
    rect_min.y = max(0, min(rect_min.y, grid.y - 1));
    rect_max.y = max(0, min(rect_max.y, grid.y - 1));

}


__forceinline__ __device__ float calculate_exponent(const glm::vec3& pix_pos,
                                                    const glm::vec3& gau_pos,
                                                    const glm::vec3& sphere_center,
                                                    const float radius,
                                                    const glm::mat3& invCovMatrix
                                                    )
{
    glm::vec3 direction_vector = gau_pos - sphere_center;

    float dir_length = glm::length(direction_vector);

    glm::vec3 normalized_vector = direction_vector / dir_length;

    glm::vec3 intersection_point = sphere_center + radius * normalized_vector;

    glm::vec3 diff = pix_pos - intersection_point;

    glm::vec3 tmp = invCovMatrix * diff;

    float power = -0.5f * glm::dot(diff, tmp);

	return power;

}


__forceinline__ __device__ void calculate_exponent_grad_v3(const glm::vec3& pix_pos,
                                                           const glm::vec3& gau_pos,
                                                           const glm::vec3& sphere_center,
                                                           const float radius,
                                                           const glm::mat3& invCovMatrix,
                                                           float dL_d_power,
                                                           glm::vec3& dL_d_gau_pos,
                                                           glm::mat3& dL_d_invCovMatrix
                                                           )
{
    // Compute the direction vector from sphere_center to gau_pos
    glm::vec3 direction_vector = gau_pos - sphere_center;

    float dir_length = glm::length(direction_vector);

    glm::vec3 normalized_vector = direction_vector / dir_length;

    glm::vec3 intersection_point = sphere_center + radius * normalized_vector;

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

    // Diagonal elements: gradient is direct
    dL_dcov3d[0] = dL_d_Cov[0][0];
    dL_dcov3d[3] = dL_d_Cov[1][1];
    dL_dcov3d[5] = dL_d_Cov[2][2];

    // Off-diagonal: each stored value maps to two symmetric positions in the full matrix,
    // so the gradient must account for both entries
    dL_dcov3d[1] = dL_d_Cov[0][1] + dL_d_Cov[1][0];
    dL_dcov3d[2] = dL_d_Cov[0][2] + dL_d_Cov[2][0];
    dL_dcov3d[4] = dL_d_Cov[1][2] + dL_d_Cov[2][1];
}


__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
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
