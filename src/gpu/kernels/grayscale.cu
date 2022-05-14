#include "helpers.cuh"
#include "../kernels.cuh"

__global__ void grayscale_gpu_kernel(unsigned char *input_buffer, int width, int height, float *output_buffer)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= height || j >= width)
    return;

  auto index = i * width + j;

  float r = input_buffer[index * 4 + 0];
  float g = input_buffer[index * 4 + 1];
  float b = input_buffer[index * 4 + 2];

  output_buffer[index] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

MatrixGPU grayscale_gpu(thrust::device_vector<unsigned char> &input, int width, int height)
{
  unsigned char *input_buffer_raw = thrust::raw_pointer_cast(input.data());

  MatrixGPU output(height, width);
  float *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  grayscale_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(input_buffer_raw, width, height, output_buffer_raw);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}
