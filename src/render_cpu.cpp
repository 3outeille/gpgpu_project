#include "render.hpp"
#include "matrix.hpp"
#include <iostream>
#include <png.h>
#include <cmath>
#include <cstddef>
#include <cassert>

Matrix grayscale(const unsigned char *image, const int &width, const int &height)
{
    auto res = Matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            auto index = i * width + j;
            double r = image[index * 4 + 0];
            double g = image[index * 4 + 1];
            double b = image[index * 4 + 2];

            // assert(image[index * 4 + 3] != 255);

            res.data[index] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.;
        }
    }

    return res;
}

Matrix gauss_kernel(const int &size)
{
    int kernel_size = 2 * size + 1;

    auto res = Matrix(kernel_size, kernel_size);

    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            auto x = j - size;
            auto y = i - size;
            auto left_x = pow(x, 2) / (2. * pow((1. / 3.) * size, 2));
            auto right_y = pow(y, 2) / (2. * pow((1. / 3.) * size, 2));
            res.data[i * kernel_size + j] = exp(-(left_x + right_y));
        }
    }

    return res;
}

Matrix compute_gradient(const Matrix &kernel, const int &size, const int &axis)
{
    /*
        axis = 1 => differentiate against y to get x
        axis = 0 => differentiate against x to get y
    */

    int kernel_size = 2 * size + 1;

    auto res = Matrix(kernel_size, kernel_size);

    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            auto previous_index = (axis == 0) ? ((i - 1) * kernel_size + j) : (i * kernel_size + j - 1);
            auto current_index = (i * kernel_size + j);
            auto next_index = (axis == 0) ? ((i + 1) * kernel_size + j) : (i * kernel_size + j + 1);

            auto local_index = (axis == 0 ? i : j);

            if (local_index == 0)
            {
                res.data[current_index] = (kernel.data[next_index] - kernel.data[current_index]);
            }
            else if (local_index == kernel_size - 1)
            {
                res.data[current_index] = (kernel.data[current_index] - kernel.data[previous_index]);
            }
            else
            {
                res.data[current_index] = (kernel.data[next_index] - kernel.data[previous_index]) / 2.;
            }
        }
    }

    return res;
}

Matrix convolution_2D(const Matrix &image, Matrix &kernel, const int &size, const int padding)
{
    int kernel_size = 2 * size + 1;
    int new_height = image.height + 2*padding - kernel_size + 1;
    int new_width = image.width + 2*padding - kernel_size + 1;

    auto res = Matrix(new_height, new_width);

    for (int i = 0; i < new_height; ++i)
    {
        for (int j = 0; j < new_width; ++j)
        {
            for (int k_i = 0; k_i < kernel_size; ++k_i)
            {
                for (int k_j = 0; k_j < kernel_size; ++k_j)
                {
                    res.data[i * new_width + j] += kernel.data[k_i * kernel_size + k_j] * image.data[(i + k_i) * image.width + (j + k_j)];
                }
            }
        }
    }

    return res;
}

Matrix gauss_derivative(const Matrix &image, const int &size, const int &axis)
{
    auto gradient = compute_gradient(gauss_kernel(size), size, axis);
    return convolution_2D(image, gradient, size, 3);
}

Matrix compute_harris_response(const Matrix &image)
{
    int size = 3;

    auto img_x = gauss_derivative(image, size, 1);
    auto img_y = gauss_derivative(image, size, 0);
    
    auto gauss = gauss_kernel(size);

    auto W_xx = convolution_2D(img_x * img_x, gauss, size, 3);
    auto W_xy = convolution_2D(img_x * img_y, gauss, size, 3);
    auto W_yy = convolution_2D(img_y * img_y, gauss, size, 3);

    auto W_det = (W_xx * W_yy) - (W_xy * W_xy);
    auto W_trace = W_xx + W_yy;

    return W_det / (W_trace + 1.);
}

std::unique_ptr<unsigned char[]> render_harris_cpu(unsigned char *buffer, int width, int height)
{
    auto image = grayscale(buffer, width, height);

    image.print_size();
    auto res_double = compute_harris_response(image);

    // auto res = image.to_buffer();

    res_double.print_size();

    return res_double.to_buffer();

    // int size = 1;
    // auto *gx = gauss_derivative_kernels(size, 1);
    // auto *gy = gauss_derivative_kernels(size, 0);

    // print_matrix(gx, 2 * size + 1);
    // print_matrix(gy, 2 * size + 1);

    /*
    gx =
        [[ 0.01098559  0.         -0.01098559]
        [ 0.988891    0.         -0.988891  ]
        [ 0.01098559  0.         -0.01098559]]

    gy =
        [[ 0.01098559  0.988891    0.01098559]
        [ 0.          0.          0.        ]
        [-0.01098559 -0.988891   -0.01098559]]
    */
}

void render_cpu(char *buffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
    std::cout << "Hello" << std::endl;

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            for (int k = 0; k < 4; k++)
            {
                buffer[i * stride + j * 4 + k] = 255;
            }
        }
    }
    return;
}
