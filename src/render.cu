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

__global__ void grayscale_kernel(unsigned char *input_buffer, int width, int height, double *output_buffer)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  auto index = i * width + j;
  double r = input_buffer[index * 4 + 0];
  double g = input_buffer[index * 4 + 1];
  double b = input_buffer[index * 4 + 2];

  output_buffer[index] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.;
}

std::unique_ptr<unsigned char[]> to_buffer(const thrust::host_vector<double> &output_buffer_host, const int &width, const int &height)
{
  auto res = std::make_unique<unsigned char[]>(width * height * 4);

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      int index = i * width + j;
      double value = std::min(std::max(0., output_buffer_host[index]), 1.);

      res.get()[index * 4 + 0] = static_cast<unsigned char>(value * 255.);
      res.get()[index * 4 + 1] = static_cast<unsigned char>(value * 255.);
      res.get()[index * 4 + 2] = static_cast<unsigned char>(value * 255.);
      res.get()[index * 4 + 3] = static_cast<unsigned char>(255.);
    }
  }

  return res;
}

void grayscale_gpu(thrust::device_vector<unsigned char> &input_buffer_device, thrust::device_vector<double> &output_buffer_device, int input_width, int input_height)
{
  unsigned char *input_buffer_raw = thrust::raw_pointer_cast(input_buffer_device.data());
  double *output_buffer_raw = thrust::raw_pointer_cast(output_buffer_device.data());

  int bsize = 32;
  int w = std::ceil((double)input_width / bsize);
  int h = std::ceil((double)input_height / bsize);

  spdlog::info("running kernel of size ({},{})", h, w);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(w, h);

  grayscale_kernel<<<dimGrid, dimBlock>>>(input_buffer_raw, input_width, input_height, output_buffer_raw);
  cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");
}

std::unique_ptr<unsigned char[]> render_harris_gpu(unsigned char *input_buffer, int input_width, int input_height, std::ptrdiff_t stride, int n_iterations)
{
  thrust::host_vector<unsigned char> input_buffer_host(input_buffer, input_buffer + (input_height * input_width * 4));
  thrust::device_vector<unsigned char> input_buffer_device = input_buffer_host;
  thrust::device_vector<double> output_buffer_device(input_height * input_width * 4);

  grayscale_gpu(input_buffer_device, output_buffer_device, input_width, input_height);

  thrust::host_vector<double> output_buffer_host = output_buffer_device;
  return to_buffer(output_buffer_host, input_width, input_height);
}

void render(char *hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  std::cout << "HELLO Harris GPU" << std::endl;
}