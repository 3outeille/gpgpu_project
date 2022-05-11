#include "render.hpp"
#include <iostream>
#include <png.h>
#include <cmath>

void print_matrix(double *mat, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cout << mat[i * size + j] << " | ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

double *gauss_kernel(int size)
{
    int kernel_size = 2 * size + 1;
    
    auto *res = (double *)calloc(kernel_size * kernel_size, sizeof(double));

    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            auto x = j - size;
            auto y = i - size;
            auto left_x = pow(x, 2) / (2. * pow((1. / 3.) * size, 2));
            auto right_y = pow(y, 2) / (2. * pow((1. / 3.) * size, 2));
            res[i * kernel_size + j] = exp(-(left_x + right_y));
        }
    }

    return res;
}

double *compute_gradient(double *kernel, int size, int axis)
{
    /*
        axis = 1 => differentiate against y to get x
        axis = 0 => differentiate against x to get y
    */

    int kernel_size = 2 * size + 1;

    auto *res = (double *)calloc(kernel_size * kernel_size, sizeof(double));

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
                res[current_index] = (kernel[next_index] - kernel[current_index]);
            }
            else if (local_index == kernel_size - 1)
            {
                res[current_index] = (kernel[current_index] - kernel[previous_index]);
            }
            else
            {
                res[current_index] = (kernel[next_index] - kernel[previous_index]) / 2.;
            }
        }
    }

    return res;
}

double *gauss_derivative_kernels(int size, int axis)
{
    return compute_gradient(gauss_kernel(size), size, axis);
}

void render_cpu(char *buffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
    int size = 1;

    auto *gx = gauss_derivative_kernels(size, 1);
    auto *gy = gauss_derivative_kernels(size, 0);

    print_matrix(gx, 2 * size + 1);
    print_matrix(gy, 2 * size + 1);

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

    free(gx);
    free(gy);

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