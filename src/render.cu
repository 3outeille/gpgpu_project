#include "render.hpp"
#include "matrix.cuh"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cstdio>

[[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void grayscale_gpu_kernel(unsigned char *input_buffer, int width, int height, double *output_buffer)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= height || j >= width)
    return;

  auto index = i * width + j;

  double r = input_buffer[index * 4 + 0];
  double g = input_buffer[index * 4 + 1];
  double b = input_buffer[index * 4 + 2];

  output_buffer[index] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

MatrixGPU grayscale_gpu(thrust::device_vector<unsigned char> &input, int width, int height)
{
  unsigned char *input_buffer_raw = thrust::raw_pointer_cast(input.data());

  MatrixGPU output(height, width);
  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  grayscale_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(input_buffer_raw, width, height, output_buffer_raw);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

__global__ void gauss_filter_gpu_kernel(double *output_buffer, int kernel_size, int size)
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

  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  gauss_filter_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, kernel_size, size);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

__global__ void compute_gradient_gpu_kernel(double *output_buffer, const double *kernel, int kernel_size, int size, int axis)
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

MatrixGPU compute_gradient_gpu(MatrixGPU input, int size, int axis)
{
  int kernel_size = 2 * size + 1;

  MatrixGPU output(kernel_size, kernel_size);

  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  compute_gradient_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, input.to_device_buffer(), kernel_size, size, axis);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

__global__ void convolution_2D_gpu_kernel(double *output_buffer, const double *input, int width, int height, const double *kernel, int kernel_size, int size)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= height || j >= width)
    return;

  double cell_value = 0;

  for (int k_i = -size; k_i <= size && i + k_i < height; ++k_i)
  {
    if (i + k_i < 0)
      continue;
    for (int k_j = -size; k_j <= size && j + k_j < width; ++k_j)
    {
      if (j + k_j < 0)
        continue;
      double image_value = input[(i + k_i) * width + (j + k_j)];
      auto kernel_value = kernel[(k_i + size) * kernel_size + (k_j + size)];
      cell_value += image_value * kernel_value;
    }
  }
  output_buffer[i * width + j] = cell_value;
}

MatrixGPU convolution_2D_gpu(MatrixGPU input, MatrixGPU kernel)
{
  int size = (kernel.width - 1) / 2;
  MatrixGPU output(input.height, input.width);

  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  convolution_2D_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, input.to_device_buffer(), input.width, input.height, kernel.to_device_buffer(), kernel.width, size);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

MatrixGPU gauss_derivative_gpu(const MatrixGPU &image, const int &size, const int &axis)
{
  auto gradient = compute_gradient_gpu(gauss_filter_gpu(size), size, axis);
  return convolution_2D_gpu(image, gradient);
}

MatrixGPU compute_harris_response_gpu(const MatrixGPU &image)
{
  int size = 3;
  auto img_x = gauss_derivative_gpu(image, size, 1);
  auto img_y = gauss_derivative_gpu(image, size, 0);

  auto gauss = gauss_filter_gpu(size);

  auto W_xx = convolution_2D_gpu(img_x * img_x, gauss);
  auto W_xy = convolution_2D_gpu(img_x * img_y, gauss);
  auto W_yy = convolution_2D_gpu(img_y * img_y, gauss);

  auto W_det = (W_xx * W_yy) - (W_xy * W_xy);
  auto W_trace = W_xx + W_yy;

  return W_det / (W_trace + 1.);
}

__global__ void circle_filter_gpu_kernel(double *output_buffer, int size)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= size || j >= size)
    return;

  auto y = static_cast<double>(i) + 0.5;
  auto x = static_cast<double>(j) + 0.5;
  auto radius = static_cast<double>(size) / 2;
  auto distance = sqrt(pow(x - radius, 2) + pow(y - radius, 2));
  output_buffer[i * size + j] = distance < radius;
}

MatrixGPU circle_filter_gpu(int size)
{
  MatrixGPU output(size, size);

  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  circle_filter_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, size);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

__global__ void morph_apply_gpu_kernel(double *output_buffer, const double *input, int width, int height, const double *kernel, int kernel_size, int size, int mode)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= height || j >= width)
    return;

  auto half_kernel = kernel_size / 2;

  double value = mode == 0 ? 1.0 : 0.0;
  for (int k_i = 0; k_i < kernel_size; k_i++)
  {
    for (int k_j = 0; k_j < kernel_size; k_j++)
    {
      if (kernel[k_i * kernel_size + k_j] == 0.)
        continue;

      auto img_i = i + k_i - half_kernel;
      auto img_j = j + k_j - half_kernel;
      double img_value = 0;

      if (!(img_i < 0 || img_i >= height || img_j < 0 || img_j >= width))
      {
        img_value = input[img_i * width + img_j];
      }

      value = mode == 0 ? fmin(value, img_value) : fmax(value, img_value);
    }
  }
  output_buffer[i * width + j] = value;
}

MatrixGPU morph_apply_gpu(MatrixGPU input, MatrixGPU kernel, int mode)
{
  // mode => erode: 0, dilate: 1
  int size = (kernel.width - 1) / 2;
  MatrixGPU output(input.height, input.width);

  double *output_buffer_raw = thrust::raw_pointer_cast(output.data.data());

  morph_apply_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output_buffer_raw, input.to_device_buffer(), input.width, input.height, kernel.to_device_buffer(), kernel.width, size, mode);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  return output;
}

std::unique_ptr<unsigned char[]> render_harris_gpu(unsigned char *input_buffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  thrust::host_vector<unsigned char> input_host(input_buffer, input_buffer + (height * width * 4));
  thrust::device_vector<unsigned char> input_device = input_host;

  spdlog::debug("Compute grayscale gpu ...");
  auto img_grayscale = grayscale_gpu(input_device, width, height);

  spdlog::debug("Compute Harris response gpu ...");
  auto harris_res = compute_harris_response_gpu(img_grayscale);

  auto image_mask = img_grayscale > 0;

  spdlog::debug("Erode shape gpu ...");
  auto min_distance = 25;
  auto eroded_mask = morph_apply_gpu(image_mask, circle_filter_gpu(min_distance * 2), 0);
  auto thresholded_mask = eroded_mask * (harris_res > (0.5 * harris_res.max()));

  spdlog::debug("Dilate Harris response...");
  auto dil = morph_apply_gpu(harris_res, circle_filter_gpu(min_distance), 1);
  auto detect_mask = thresholded_mask * (harris_res == dil);

  auto res = detect_mask * 255;
  return res.to_host_buffer();
}
