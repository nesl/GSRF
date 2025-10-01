#include "tracer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n
					 )
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;

	while (step > 1) {
		step /= 2;

		if (n >> msb) {
			msb += step;

		} else {
			msb -= step;
		}
	}

	if (n >> msb) {
		msb++;
	}
	
	return msb;
}



// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, 
								  uint32_t* geom_tiles_touched,
								  const float* geom_depths,
								  const uint32_t* geom_point_offsets,
								  const uint2* geom_rect_mins,
								  const uint2* geom_rect_maxs,
								  uint64_t* bin_point_list_keys_unsorted,
								  uint32_t* bin_point_list_unsorted,
								  dim3 tile_grid
								  )
{

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (geom_tiles_touched[idx] > 0) {

		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : geom_point_offsets[idx - 1];

		uint2 rect_min = geom_rect_mins[idx];
		uint2 rect_max = geom_rect_maxs[idx];


		for (int y = rect_min.y; y < rect_max.y + 1; y++) {

			for (int x = rect_min.x; x < rect_max.x + 1; x++) {

				uint64_t key = y * tile_grid.x + x;
				key <<= 32;

				key |= *((uint32_t*)&geom_depths[idx]);

				bin_point_list_keys_unsorted[off] = key;

				bin_point_list_unsorted[off] = idx;

				off++;

			}
		}

	}

}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int num_rendered_t, 
								   uint64_t* bin_point_list_keys, 
								   uint2* img_ranges
								   )
{	
	// point_list_keys: sorted (tileID|depth) list
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_rendered_t)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = bin_point_list_keys[idx];

	uint32_t currtile = key >> 32;

	if (idx == 0) {

		img_ranges[currtile].x = 0;

	} else {

		uint32_t prevtile = bin_point_list_keys[idx - 1] >> 32;

		if (currtile != prevtile) {

			img_ranges[prevtile].y = idx;
			img_ranges[currtile].x = idx;
		}
	}

	if (idx == num_rendered_t - 1) {
		img_ranges[currtile].y = num_rendered_t;
	}

}


CudaTracer::GeometryState CudaTracer::GeometryState::fromChunk(char*& chunk, 
																	   size_t P
																	   )
{

	GeometryState geom;

	obtain(chunk, geom.depths,   P, 128);
	obtain(chunk, geom.rec_mins, P, 128);
	obtain(chunk, geom.rec_maxs, P, 128);

	obtain(chunk, geom.means_2d, P, 128);

	obtain(chunk, geom.rgb, P * NUM_CHANNELS, 128);

	obtain(chunk, geom.clamped,  P * NUM_CHANNELS, 128);

	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);

	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);

	return geom;
}


CudaTracer::ImageState CudaTracer::ImageState::fromChunk(char*& chunk, 
																 size_t N
																 )
{

	ImageState img;

	obtain(chunk, img.accum_alpha, N, 128);

	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);

	return img;
}


CudaTracer::BinningState CudaTracer::BinningState::fromChunk(char*& chunk, 
																	 size_t P
																	 ) 
{

	BinningState binning;

	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);

	cub::DeviceRadixSort::SortPairs(nullptr, 
									binning.sorting_size,
									binning.point_list_keys_unsorted, 
									binning.point_list_keys,
									binning.point_list_unsorted, 
									binning.point_list, 
									P);

	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);

	return binning;
}



int CudaTracer::Tracer::forward(const int P,
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
										)
{

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr    = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(W * H);
	char* img_chunkptr    = imageBuffer(img_chunk_size);
	ImageState imgState   = ImageState::fromChunk(img_chunkptr, W * H);

	// Run preprocessing per-Gaussian (transformation, bounding)
	CHECK_CUDA(FORWARD::preprocess(P,
								   means_3d,
								   cov3d_precomp,
								   signal_precomp,
								   gaus_radii,
								   H,
								   W,
								   spectrum_3d_coarse,
								   (glm::vec3*)rx_pos,
								   radius_rx,
								   tx_pos,
								   sh_degree_active,
								   sh_coef_len_max, 
								   geomState.depths,
								   geomState.tiles_touched,
								   geomState.rec_mins,
								   geomState.rec_maxs,
								   geomState.means_2d,
								   geomState.rgb,
								   geomState.clamped,
								   tile_grid
								   ), debug);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, 
											 geomState.scan_size, 
											 geomState.tiles_touched, 
											 geomState.point_offsets, 
											 P
											 ), debug);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr    = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (P, 
													 geomState.tiles_touched, 
													 geomState.depths,
													 geomState.point_offsets,
													 geomState.rec_mins,
													 geomState.rec_maxs,
													 binningState.point_list_keys_unsorted,
													 binningState.point_list_unsorted,
													 tile_grid
													 );
	CHECK_CUDA(, debug);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space,
											   binningState.sorting_size,
											   binningState.point_list_keys_unsorted, 
											   binningState.point_list_keys,
											   binningState.point_list_unsorted, 
											   binningState.point_list,
											   num_rendered, 
											   0, 
											   32 + bit
											   ), debug);

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);


	// Identify start and end of per-tile workloads in sorted gaussian ID (point_list)
	if (num_rendered > 0) {
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (num_rendered,
																	 binningState.point_list_keys,  // (tileID | depth) key list
																	 imgState.ranges
																	 );
		CHECK_CUDA(, debug);
	}

	CHECK_CUDA(FORWARD::render(tile_grid,
							   block,
							   means_3d,
							   cov3d_precomp,
							   attenuation,
							   gaus_radii,
							   H,
							   W,
							   sh_degree_active,
							   sh_coef_len_max,
							   spectrum_3d_fine,
							   (glm::vec3*)rx_pos,
							   radius_rx,
							   background,
							   binningState.point_list,
							   imgState.ranges,
							   geomState.means_2d,
							   geomState.rgb,
							   imgState.accum_alpha,  
							   imgState.n_contrib,
							   out_color
							   ), debug);

	return num_rendered;
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaTracer::Tracer::backward(const int P,
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
										  )
{
	GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
	ImageState imgState       = ImageState::fromChunk(image_buffer, W * H);

	const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	CHECK_CUDA(BACKWARD::render(tile_grid,
								block,
								dL_dout_color,
								means_3d,
								cov3d_precomp,
								attenuation,
								gaus_radii,
								H,
								W,
								sh_degree_active,
								sh_coef_len_max,
								spectrum_3d_fine,
								(glm::vec3*)rx_pos,
								radius_rx,
								background,
								geomState.rgb,
								binningState.point_list,
								imgState.ranges,
								imgState.accum_alpha,
								imgState.n_contrib,
								grad_means_3d, 
								grad_cov3d_precomp,
								grad_attenuation,
								dL_dcolor,
								debug
								), debug);

}




