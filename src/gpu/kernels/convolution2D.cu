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

MatrixGPU convolution_2D_gpu(MatrixGPU &input, MatrixGPU &kernel)
{
  int size = (kernel.width - 1) / 2;
  MatrixGPU output(input.height, input.width);

  float *input_pitched;
  size_t pitch;
  cudaMallocPitch(&input_pitched, &pitch, input.width * sizeof(float), input.height);
  cudaMemcpy2D(input_pitched, pitch, input.to_device_buffer(), input.width * sizeof(float), input.width * sizeof(float), input.height, cudaMemcpyDeviceToDevice);

  convolution_2D_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output.to_device_buffer(), input_pitched, input.width, input.height, pitch, kernel.to_device_buffer(), kernel.width, size);
  cudaDeviceSynchronize();

  cudaFree(input_pitched);

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}
