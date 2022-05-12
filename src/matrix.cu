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

MatrixCuDevice::MatrixCuDevice()
    : height(1), width(1)
{
    data_device = thrust::device_vector<double>(height * width * 4);
}

MatrixCuDevice::MatrixCuDevice(unsigned char *buffer, int height, int width)
    : height(height), width(width)
{
    thrust::host_vector<double> data_host(height * width * 4);

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            data_host[i * width + j] = buffer[i * width + j];
        }
    }

    data_device = data_host;
}

MatrixCuDevice::MatrixCuDevice(int height, int width)
    : height(height), width(width)
{
    data_device = thrust::device_vector<double>(height * width * 4);
}

void MatrixCuDevice::print()
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << data_device[i * width + j] << " | ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void MatrixCuDevice::print_size()
{
    std::cout << width << "x" << height << std::endl;
}

std::unique_ptr<unsigned char[]> MatrixCuDevice::to_buffer()
{
    // Convert device to host 
    thrust::host_vector<double> data_host = data_device;

    auto res = std::make_unique<unsigned char[]>(width * height * 4);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            double value = std::min(std::max(0., data_host[index]), 1.);

            res.get()[index * 4 + 0] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 1] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 2] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 3] = static_cast<unsigned char>(255.);
        }
    }

    return res;
}

thrust::host_vector<double> MatrixCuDevice::to_host() {
    thrust::host_vector<double> data_host = data_device;
    return data_host;
}
