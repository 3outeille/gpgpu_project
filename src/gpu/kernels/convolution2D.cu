#include "../kernels.cuh"
#include "helpers.cuh"

__global__ void convolution_2D_gpu_kernel(float *output_buffer, const float *input, int width, int height, size_t pitch, const float *kernel, int kernel_size, int size)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= height || j >= width)
		return;

	float cell_value = 0;

	for (int k_i = -size; k_i <= size && i + k_i < height; ++k_i)
	{
		if (i + k_i < 0)
			continue;
		for (int k_j = -size; k_j <= size && j + k_j < width; ++k_j)
		{
			if (j + k_j < 0)
				continue;
			float image_value = *eltPtr(input, j + k_j, i + k_i, pitch);
			auto kernel_value = kernel[(k_i + size) * kernel_size + (k_j + size)];
			cell_value += image_value * kernel_value;
		}
	}
	output_buffer[i * width + j] = cell_value;
}

const int BLOCK_SIZE = 32;

__global__ void convolution_2D_tiled_gpu_kernel(float *output_buffer, const float *input, int width, int height, size_t pitch, const float *kernel, int kernel_size, int size)
{
	__shared__ float padded_tile[BLOCK_SIZE][BLOCK_SIZE];

	int tile_size = BLOCK_SIZE - kernel_size + 1;

	int input_i = threadIdx.y + blockIdx.y * tile_size;
	int input_j = threadIdx.x + blockIdx.x * tile_size;

	// Load tile
	int block_i = input_i - (kernel_size / 2);
	int block_j = input_j - (kernel_size / 2);

	if (block_i >= 0 && block_i < height && block_j >= 0 && block_j < width)
		padded_tile[threadIdx.y][threadIdx.x] = *eltPtr(input, block_j, block_i, pitch);
	else
		padded_tile[threadIdx.y][threadIdx.x] = 0.0f;

	__syncthreads();

	if (threadIdx.x >= tile_size || threadIdx.y >= tile_size || input_i >= height || input_j >= width)
		return;

	float cell_value = 0.;

	for (int k_i = 0; k_i < kernel_size; ++k_i)
		for (int k_j = 0; k_j < kernel_size; ++k_j)
			cell_value += padded_tile[threadIdx.y + k_i][threadIdx.x + k_j] * kernel[k_i * kernel_size + k_j];

	output_buffer[input_i * width + input_j] = cell_value;
}

MatrixGPU convolution_2D_gpu(MatrixGPU &input, MatrixGPU &kernel)
{
	int size = (kernel.width - 1) / 2;

	float *input_pitched;
	size_t pitch;
	cudaMallocPitch(&input_pitched, &pitch, input.width * sizeof(float), input.height);
	cudaMemcpy2D(input_pitched, pitch, input.to_device_buffer(), input.width * sizeof(float), input.width * sizeof(float), input.height, cudaMemcpyDeviceToDevice);


	MatrixGPU output(input.height, input.width);

	// convolution_2D_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output.to_device_buffer(), input_pitched, input.width, input.height, pitch, kernel.to_device_buffer(), kernel.width, size);

	auto dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	float tile_size = BLOCK_SIZE - kernel.width + 1;
	auto dim_grid_width = std::ceil((float)input.width / tile_size);
	auto dim_grid_height = std::ceil((float)input.height / tile_size);
	auto dim_grid = dim3(dim_grid_width, dim_grid_height);
	convolution_2D_tiled_gpu_kernel<<<dim_grid, dim_block>>>(output.to_device_buffer(), input_pitched, input.width, input.height, pitch, kernel.to_device_buffer(), kernel.width, size);

	cudaDeviceSynchronize();

	if (cudaPeekAtLastError())
		abortError("Computation Error");

	cudaFree(input_pitched);

	return output;
}
