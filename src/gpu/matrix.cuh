#pragma once
#include <memory>
#include <cstddef>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct MatrixGPU
{
    MatrixGPU(float *buffer, int height, int width);
    MatrixGPU(int height, int width);

    MatrixGPU operator*(const MatrixGPU &rhs) const;
    MatrixGPU operator*(const float &rhs) const;
    MatrixGPU operator+(const MatrixGPU &rhs) const;
    MatrixGPU operator+(const float &rhs) const;
    MatrixGPU operator-(const MatrixGPU &rhs) const;
    MatrixGPU operator/(const MatrixGPU &rhs) const;
    MatrixGPU operator>(const float &rhs) const;
    MatrixGPU operator==(const MatrixGPU &rhs) const;
    float max();

    std::unique_ptr<unsigned char[]> to_host_buffer();
    float *to_device_buffer();

    dim3 dimBlock() const;
    dim3 dimGrid() const;

    void display() const;
    void print_size() const;

    thrust::device_vector<float> data;
    int height, width;
    const int bsize = 32;
};
