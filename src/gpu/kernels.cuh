#pragma once

#include "matrix.cuh"

MatrixGPU grayscale_gpu(thrust::device_vector<unsigned char> &input, int width, int height);

MatrixGPU gauss_filter_gpu(int size);
MatrixGPU sobel_filter_gpu(MatrixGPU &image, const int &axis);
MatrixGPU convolution_2D_gpu(MatrixGPU &input, MatrixGPU &kernel);

std::tuple<thrust::device_vector<char>, int> circle_filter_gpu(int size);
MatrixGPU morph_apply_gpu(MatrixGPU &input, const std::tuple<thrust::device_vector<char>, int> &kernel, int mode);
