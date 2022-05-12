#include "render.hpp"
#include "matrix.cuh"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>

[[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

void print_matrix(int *v1, int width, int height)
{
  std::cout << "---------------------" << std::endl;
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      std::cout << v1[i * width + j] << " | ";
    }
    std::cout << std::endl;
  }
}

// Device code
__global__ void mykernel(char *buffer, int width, int height, size_t pitch)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < width && col < height)
  {
    *(((int *)(((char *)buffer) + (row * pitch))) + col) = 9;
  }
}

std::unique_ptr<unsigned char[]> render_harris_gpu(char *hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  auto res = MatrixCu(2, 2);
  std::cout << res.data.size() << std::endl;

  thrust::fill(res.data.begin(), res.data.end(), 42);

  res.print();

  return res.to_buffer();

  // int width = 4;
  // int height = 4;

  // cudaError_t rc = cudaSuccess;

  // // Allocate device memory
  // char *devBuffer;
  // size_t pitch;

  // rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(int), height);
  // if (rc)
  //   abortError("Fail buffer allocation");

  // // Run the kernel with blocks of size 64 x 64
  // {
  //   int bsize = 1;
  //   int w = std::ceil((float)width / bsize);
  //   int h = std::ceil((float)height / bsize);

  //   spdlog::debug("running kernel of size ({},{})", w, h);

  //   dim3 dimBlock(bsize, bsize);
  //   dim3 dimGrid(w, h);
  //   mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

  //   if (cudaPeekAtLastError())
  //     abortError("Computation Error");
  // }

  // // Copy back to main memory
  // rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost);
  // if (rc)
  //   abortError("Unable to copy buffer back to memory");

  // // Free
  // rc = cudaFree(devBuffer);
  // if (rc)
  //   abortError("Unable to free memory");

  // print_matrix(hostBuffer, width, height);
  // printf("\nPitch size: %ld \n", pitch);
}

void render(char *hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  std::cout << "HELLO Harris GPU" << std::endl;
}