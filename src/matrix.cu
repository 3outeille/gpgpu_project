#include "matrix.cuh"
#include <memory>
#include <iostream>
#include <cstddef>
#include <cstddef>
#include <memory>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

MatrixGPU::MatrixGPU(double *buffer, int height, int width)
    : height(height), width(width)
{
    thrust::host_vector<double> data_host(height * width * 4);

    for (int i = 0; i < width * height; ++i)
        data_host[i] = buffer[i];

    data = data_host;
}

MatrixGPU::MatrixGPU(int height, int width)
    : height(height), width(width), data(height * width * 4)
{
}

MatrixGPU::MatrixGPU(thrust::device_vector<double> vec, int height, int width)
    : height(height), width(width), data(vec)
{
}

MatrixGPU::MatrixGPU(thrust::host_vector<double> vec, int height, int width)
    : height(height), width(width), data(vec)
{
}

void MatrixGPU::print_size()
{
    std::cout << width << "x" << height << std::endl;
}

void MatrixGPU::display()
{
    thrust::host_vector<double> data_host = data;

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << data_host[i * width + j] << " | ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

std::unique_ptr<unsigned char[]> MatrixGPU::to_host_buffer()
{
    thrust::host_vector<double> data_host = data;

    auto res = std::make_unique<unsigned char[]>(width * height * 4);

    for (int i = 0; i < width * height; i++)
    {
        double value = std::min(std::max(0., data_host[i]), 1.);

        res.get()[i * 4 + 0] = static_cast<unsigned char>(value * 255.);
        res.get()[i * 4 + 1] = static_cast<unsigned char>(value * 255.);
        res.get()[i * 4 + 2] = static_cast<unsigned char>(value * 255.);
        res.get()[i * 4 + 3] = static_cast<unsigned char>(255.);
    }

    return res;
}

double *MatrixGPU::to_device_buffer()
{
    return thrust::raw_pointer_cast(data.data());
}

dim3 MatrixGPU::dimBlock()
{
    return dim3(bsize, bsize);
}

dim3 MatrixGPU::dimGrid()
{
    int w = std::ceil((double)width / bsize);
    int h = std::ceil((double)height / bsize);
    return dim3(w, h);
}

MatrixGPU MatrixGPU::operator*(const MatrixGPU &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::multiplies<double>());

    return res;
}

MatrixGPU MatrixGPU::operator*(const double &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        thrust::make_constant_iterator(rhs),
        res.data.begin(),
        thrust::multiplies<double>());

    return res;
}

MatrixGPU MatrixGPU::operator+(const MatrixGPU &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::plus<double>());

    return res;
}

MatrixGPU MatrixGPU::operator+(const double &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        thrust::make_constant_iterator(rhs),
        res.data.begin(),
        thrust::plus<double>());

    return res;
}

MatrixGPU MatrixGPU::operator-(const MatrixGPU &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::minus<double>());

    return res;
}

MatrixGPU MatrixGPU::operator/(const MatrixGPU &rhs)
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::divides<double>());

    return res;
}

MatrixGPU MatrixGPU::operator>(const double &rhs)
{
    std::cout << "TODO: IMPLEMENT > operator" << std::endl;
    return MatrixGPU(height, width);
}

MatrixGPU MatrixGPU::operator==(const MatrixGPU &rhs)
{
    std::cout << "TODO: IMPLEMENT == operator" << std::endl;
    return MatrixGPU(height, width);
}
