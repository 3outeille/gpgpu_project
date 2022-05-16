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

MatrixGPU::MatrixGPU(float *buffer, int height, int width)
    : height(height), width(width)
{
    thrust::host_vector<float> data_host(height * width);

    for (int i = 0; i < width * height; ++i)
        data_host[i] = buffer[i];

    data = data_host;
}

MatrixGPU::MatrixGPU(int height, int width)
    : height(height), width(width), data(height * width)
{
}

void MatrixGPU::print_size() const
{
    std::cout << width << "x" << height << std::endl;
}

void MatrixGPU::display() const
{
    thrust::host_vector<float> data_host = data;

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
    thrust::host_vector<float> data_host = data;

    auto res = std::make_unique<unsigned char[]>(width * height * 4);

    for (int i = 0; i < width * height; i++)
    {
        float value = std::fmin(std::fmax(0., data_host[i]), 255.);

        res.get()[i * 4 + 0] = static_cast<unsigned char>(value);
        res.get()[i * 4 + 1] = static_cast<unsigned char>(value);
        res.get()[i * 4 + 2] = static_cast<unsigned char>(value);
        res.get()[i * 4 + 3] = static_cast<unsigned char>(255.);
    }

    return res;
}

float *MatrixGPU::to_device_buffer()
{
    return thrust::raw_pointer_cast(data.data());
}

dim3 MatrixGPU::dimBlock() const
{
    return dim3(bsize, bsize);
}

dim3 MatrixGPU::dimGrid() const
{
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);
    return dim3(w, h);
}

MatrixGPU MatrixGPU::operator*(const MatrixGPU &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::multiplies<float>());

    return res;
}

MatrixGPU MatrixGPU::operator*(const float &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        thrust::make_constant_iterator(rhs),
        res.data.begin(),
        thrust::multiplies<float>());

    return res;
}

MatrixGPU MatrixGPU::operator+(const MatrixGPU &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::plus<float>());

    return res;
}

MatrixGPU MatrixGPU::operator+(const float &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        thrust::make_constant_iterator(rhs),
        res.data.begin(),
        thrust::plus<float>());

    return res;
}

MatrixGPU MatrixGPU::operator-(const MatrixGPU &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::minus<float>());

    return res;
}

MatrixGPU MatrixGPU::operator/(const MatrixGPU &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::divides<float>());

    return res;
}

MatrixGPU MatrixGPU::operator>(const float &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        thrust::make_constant_iterator(rhs),
        res.data.begin(),
        thrust::greater<float>());

    return res;
}

MatrixGPU MatrixGPU::operator==(const MatrixGPU &rhs) const
{
    MatrixGPU res(height, width);

    thrust::transform(
        data.begin(),
        data.end(),
        rhs.data.begin(),
        res.data.begin(),
        thrust::equal_to<float>());

    return res;
}

float MatrixGPU::max()
{
    return *thrust::max_element(data.begin(), data.end());
}
