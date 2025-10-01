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

#define NUM_CHANNELS 2
#define BLOCK_X 16
#define BLOCK_Y 16


#define MAX_FREQ_LOG2 9
#define NUM_FREQS 10

#define INPUT_DIM_DIR 3

#define EMBEDDING_DIM (INPUT_DIM_DIR + 2 * INPUT_DIM_DIR * NUM_FREQS)

#define INPUT_DIM_EMD (EMBEDDING_DIM * 2)

#define HIDDEN_DIM_1 256
#define HIDDEN_DIM_2 64
#define OUTPUT_DIM 3
// #define TOTAL_PARAMS 20483

#define TOTAL_PARAMS (INPUT_DIM_EMD * HIDDEN_DIM_1 + HIDDEN_DIM_1 + HIDDEN_DIM_1 * HIDDEN_DIM_2 + HIDDEN_DIM_2 + HIDDEN_DIM_2 * OUTPUT_DIM + OUTPUT_DIM)


#define MAX_GRID_X 23   // 360 / 16
#define MAX_GRID_Y 6    // 90 / 16

#endif

