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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// output channels: 26 subcarriers x 2 (real + imaginary) = 52
#define NUM_CHANNELS 52  // 26 subcarriers × 2 (real + imag) for CSI
// CUDA thread block dimensions for tile-based rendering
#define BLOCK_X 16
#define BLOCK_Y 16

#endif

