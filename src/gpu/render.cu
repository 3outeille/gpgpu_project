#include "../render.hpp"
#include "matrix.cuh"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include "kernels/helpers.cuh"
#include "kernels.cuh"

MatrixGPU compute_harris_response_gpu(MatrixGPU &image)
{
  int size = 3;
  auto img_x = sobel_filter_gpu(image, size, 1);
  auto img_y = sobel_filter_gpu(image, size, 0);

  auto gauss = gauss_filter_gpu(size);

  auto I_xx = img_x * img_x;
  auto I_xy = img_x * img_y;
  auto I_yy = img_y * img_y;

  auto W_xx = convolution_2D_gpu(I_xx, gauss);
  auto W_xy = convolution_2D_gpu(I_xy, gauss);
  auto W_yy = convolution_2D_gpu(I_yy, gauss);

  auto W_det = (W_xx * W_yy) - (W_xy * W_xy);
  auto W_trace = W_xx + W_yy;

  return W_det / (W_trace + 1.);
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
