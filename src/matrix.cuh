#pragma once
#include <memory>
#include <cstddef>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class MatrixGPU
{
public:
    MatrixGPU(double *buffer, int height, int width);
    MatrixGPU(int height, int width);
    MatrixGPU(thrust::device_vector<double> vec, int height, int width);
    MatrixGPU(thrust::host_vector<double> vec, int height, int width);

    MatrixGPU operator*(const MatrixGPU &rhs);
    MatrixGPU operator*(const double &rhs);
    MatrixGPU operator+(const MatrixGPU &rhs);
    MatrixGPU operator+(const double &rhs);
    MatrixGPU operator-(const MatrixGPU &rhs);
    MatrixGPU operator/(const MatrixGPU &rhs);
    MatrixGPU operator>(const double &rhs);
    MatrixGPU operator==(const MatrixGPU &rhs);

    std::unique_ptr<unsigned char[]> to_host_buffer();
    double *to_device_buffer();

    dim3 dimBlock();
    dim3 dimGrid();

    void display();
    void print_size();

    thrust::device_vector<double> data;
    int height, width;
    const int bsize = 32;
};
