#include "../kernels.cuh"
#include "helpers.cuh"

__global__ void gauss_filter_gpu_kernel(float *output_buffer, int kernel_size, int size)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= kernel_size || j >= kernel_size)
    return;

  auto x = j - size;
  auto y = i - size;
  auto left_x = pow(x, 2) / (2. * pow((1. / 3.) * size, 2));
  auto right_y = pow(y, 2) / (2. * pow((1. / 3.) * size, 2));
  output_buffer[i * kernel_size + j] = exp(-(left_x + right_y));
}

MatrixGPU gauss_filter_gpu(int size)
{
  int kernel_size = 2 * size + 1;

  MatrixGPU output(kernel_size, kernel_size);

  float *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  gauss_filter_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, kernel_size, size);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

__global__ void compute_gradient_gpu_kernel(float *output_buffer, const float *kernel, int kernel_size, int size, int axis)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= kernel_size || j >= kernel_size)
    return;

  auto previous_index = (axis == 0) ? ((i - 1) * kernel_size + j) : (i * kernel_size + j - 1);
  auto current_index = (i * kernel_size + j);
  auto next_index = (axis == 0) ? ((i + 1) * kernel_size + j) : (i * kernel_size + j + 1);

  auto local_index = (axis == 0 ? i : j);

  if (local_index == 0)
  {
    output_buffer[current_index] = (kernel[next_index] - kernel[current_index]);
  }
  else if (local_index == kernel_size - 1)
  {
    output_buffer[current_index] = (kernel[current_index] - kernel[previous_index]);
  }
  else
  {
    output_buffer[current_index] = (kernel[next_index] - kernel[previous_index]) / 2.;
  }
}

MatrixGPU compute_gradient_gpu(MatrixGPU &input, int size, int axis)
{
  int kernel_size = 2 * size + 1;

  MatrixGPU output(kernel_size, kernel_size);

  float *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  compute_gradient_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, input.to_device_buffer(), kernel_size, size, axis);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

MatrixGPU sobel_filter_gpu(MatrixGPU &image, const int &size, const int &axis)
{
  auto gauss = gauss_filter_gpu(size);
  auto gradient = compute_gradient_gpu(gauss, size, axis);
  return convolution_2D_gpu(image, gradient);
}
