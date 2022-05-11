#include "matrix.hpp"
#include <memory>
#include <iostream>
#include <cstddef>
#include <cstddef>
#include <memory>
#include <cassert>
#include <iostream>

Matrix::Matrix()
    : height(1), width(1)
{
    data = (double *)calloc(height * width, sizeof(double));
}

Matrix::~Matrix()
{
    free(data);
}

Matrix::Matrix(int height, int width)
    : height(height), width(width)
{
    data = (double *)calloc(height * width, sizeof(double));
}

Matrix Matrix::operator*(const Matrix &rhs)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res.data[i * width + j] = this->data[i * width + j] * rhs.data[i * width + j];
        }
    }

    return res;
}

Matrix Matrix::operator+(const Matrix &rhs)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res.data[i * width + j] = this->data[i * width + j] + rhs.data[i * width + j];
        }
    }

    return res;
}

Matrix Matrix::operator+(const double &rhs)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res.data[i * width + j] = this->data[i * width + j] + rhs;
        }
    }

    return res;
}

Matrix Matrix::operator-(const Matrix &rhs)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res.data[i * width + j] = this->data[i * width + j] - rhs.data[i * width + j];
        }
    }

    return res;
}

Matrix Matrix::operator/(const Matrix &rhs)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res.data[i * width + j] = this->data[i * width + j] / rhs.data[i * width + j];
        }
    }

    return res;
}

void Matrix::print()
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

void Matrix::print_size()
{
    std::cout << this->width << "x" << this->height << std::endl;
}

std::unique_ptr<unsigned char[]> Matrix::to_buffer()
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
