#include "matrix.cuh"
#include <memory>
#include <iostream>
#include <cstddef>
#include <cstddef>
#include <memory>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>

MatrixCu::MatrixCu()
    : height(1), width(1)
{
    data = thrust::host_vector<double>(height * width);
}

MatrixCu::MatrixCu(int height, int width)
    : height(height), width(width)
{
    data = thrust::host_vector<double>(height * width);
}

// MatrixCu::~MatrixCu()
// {
//     // data.~host_vector();    
// }

void MatrixCu::print()
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << data[i * width + j] << " | ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void MatrixCu::print_size()
{
    std::cout << this->width << "x" << this->height << std::endl;
}

std::unique_ptr<unsigned char[]> MatrixCu::to_buffer()
{
    auto res = std::make_unique<unsigned char[]>(width * height * 4);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            double value = std::min(std::max(0., data[index]), 1.);

            res.get()[index * 4 + 0] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 1] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 2] = static_cast<unsigned char>(value * 255.);
            res.get()[index * 4 + 3] = static_cast<unsigned char>(255.);
        }
    }

    return res;
}
