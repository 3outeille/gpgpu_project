#include "matrix.hpp"
#include <memory>
#include <iostream>
#include <cstddef>
#include <cstddef>
#include <memory>
#include <cassert>
#include <iostream>

Matrix::Matrix(int height, int width)
    : height(height), width(width), data(height * width)
{
    std::fill(data.begin(), data.end(), 0);
}

Matrix Matrix::operator*(const Matrix &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] * rhs.data[i];

    return res;
}

Matrix Matrix::operator*(const double &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = data[i] * rhs;

    return res;
}

Matrix Matrix::operator+(const Matrix &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] + rhs.data[i];

    return res;
}

Matrix Matrix::operator+(const double &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] + rhs;

    return res;
}

Matrix Matrix::operator-(const Matrix &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] - rhs.data[i];

    return res;
}

Matrix Matrix::operator/(const Matrix &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] / rhs.data[i];

    return res;
}

void Matrix::print() const
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

void Matrix::print_size() const
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
            double value = static_cast<unsigned char>(std::min(std::max(0., data[index]), 255.));

            res.get()[index * 4 + 0] = value;
            res.get()[index * 4 + 1] = value;
            res.get()[index * 4 + 2] = value;
            res.get()[index * 4 + 3] = static_cast<unsigned char>(255.);
        }
    }

    return res;
}

Matrix Matrix::operator>(const double &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = this->data[i] > rhs;

    return res;
}

Matrix Matrix::operator==(const Matrix &rhs) const
{
    auto res = Matrix(height, width);

    for (int i = 0; i < width * height; i++)
        res.data[i] = data[i] == rhs.data[i];

    return res;
}

double Matrix::max() const
{
    auto res = data[0];

    for (int i = 0; i < width * height; i++)
        res = std::max(data[i], res);

    return res;
}
