#pragma once
#include <memory>
#include <cstddef>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class MatrixGPU
{
public:
    MatrixGPU(float *buffer, int height, int width);
    MatrixGPU(int height, int width);

    MatrixGPU operator*(const MatrixGPU &rhs);
    MatrixGPU operator*(const float &rhs);
    MatrixGPU operator+(const MatrixGPU &rhs);
    MatrixGPU operator+(const float &rhs);
    MatrixGPU operator-(const MatrixGPU &rhs);
    MatrixGPU operator/(const MatrixGPU &rhs);
    MatrixGPU operator>(const float &rhs);
    MatrixGPU operator==(const MatrixGPU &rhs);
    float max();

    std::unique_ptr<unsigned char[]> to_host_buffer();
    float *to_device_buffer();

    dim3 dimBlock();
    dim3 dimGrid();

    void display();
    void print_size();

    thrust::device_vector<float> data;
    int height, width;
    const int bsize = 32;
};
