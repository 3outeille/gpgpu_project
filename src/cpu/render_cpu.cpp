#include "matrix.hpp"
#include <iostream>
#include <png.h>
#include <cmath>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <cassert>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <fstream>

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

            res.data[index] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    return res;
}

Matrix gauss_filter(const int &size)
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

Matrix convolution_2D(const Matrix &image, Matrix &kernel, const int &size)
{
    int kernel_size = 2 * size + 1;
    auto res = Matrix(image.height, image.width);

    for (int i = 0; i < image.height; ++i)
    {
        for (int j = 0; j < image.width; ++j)
        {
            double cell_value = 0;

            for (int k_i = -size; k_i <= size; ++k_i)
            {
                for (int k_j = -size; k_j <= size; ++k_j)
                {
                    double image_value = 0;
                    if (!(i + k_i < 0 || i + k_i >= image.height || j + k_j < 0 || j + k_j >= image.width))
                        image_value = image.data[(i + k_i) * image.width + (j + k_j)];

                    auto kernel_value = kernel.data[(k_i + size) * kernel_size + (k_j + size)];
                    cell_value += image_value * kernel_value;
                }
            }

            res.data[i * image.width + j] = cell_value;
        }
    }

    return res;
}

Matrix gauss_derivative(const Matrix &image, const int &size, const int &axis)
{
    auto gradient = compute_gradient(gauss_filter(size), size, axis);
    return convolution_2D(image, gradient, size);
}

Matrix compute_harris_response(const Matrix &image)
{
    int size = 3;

    auto img_x = gauss_derivative(image, size, 1);
    auto img_y = gauss_derivative(image, size, 0);

    auto gauss = gauss_filter(size);

    auto W_xx = convolution_2D(img_x * img_x, gauss, size);
    auto W_xy = convolution_2D(img_x * img_y, gauss, size);
    auto W_yy = convolution_2D(img_y * img_y, gauss, size);

    auto W_det = (W_xx * W_yy) - (W_xy * W_xy);
    auto W_trace = W_xx + W_yy;

    return W_det / (W_trace + 1.);
}

Matrix morph_circle_kernel(const int kernel_size)
{
    auto res = Matrix(kernel_size, kernel_size);

    auto radius = static_cast<double>(kernel_size) / 2;

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            auto y = static_cast<double>(i) + 0.5;
            auto x = static_cast<double>(j) + 0.5;
            auto distance = sqrt(pow(x - radius, 2) + pow(y - radius, 2));
            res.data[i * kernel_size + j] = distance < radius;
        }
    }

    return res;
}

Matrix morph_apply_kernel(const Matrix &image, const Matrix &kernel, int mode)
{
    // mode => erode: 0, dilate: 1

    auto kernel_size = kernel.width;
    auto half_kernel = kernel_size / 2;

    auto res = Matrix(image.height, image.width);

    for (int i = 0; i < image.height; i++)
    {
        for (int j = 0; j < image.width; j++)
        {
            double value = mode == 0 ? 1.0 : 0.0;
            for (int k_i = 0; k_i < kernel_size; k_i++)
            {
                for (int k_j = 0; k_j < kernel_size; k_j++)
                {
                    if (kernel.data[k_i * kernel_size + k_j] == 0.)
                        continue;

                    auto img_i = i + k_i - half_kernel;
                    auto img_j = j + k_j - half_kernel;
                    double img_value = 0;

                    if (!(img_i < 0 || img_i >= image.height || img_j < 0 || img_j >= image.width))
                    {
                        img_value = image.data[(i + k_i - kernel_size / 2) * image.width + (j + k_j - kernel_size / 2)];
                    }

                    value = mode == 0 ? std::min(value, img_value) : std::max(value, img_value);
                }
            }
            res.data[i * res.width + j] = value;
        }
    }

    return res;
}

std::vector<std::tuple<int, int>> top_k_best_coords_keypoints(Matrix detect_mask, Matrix harris_res, size_t K = 10)
{
    // coordinates of candidates
    std::vector<std::tuple<int, int>> candidates_coords;
    for (int i = 0; i < detect_mask.height; ++i)
    {
        for (int j = 0; j < detect_mask.width; ++j)
        {
            if (detect_mask.data[i * detect_mask.width + j] != 0.)
                candidates_coords.push_back(std::make_tuple(i, j));
        }
    }

    // values of candidates
    std::vector<double> candidate_values;
    for (int i = 0; i < detect_mask.width * detect_mask.height; ++i)
    {
        if (detect_mask.data[i])
            candidate_values.push_back(harris_res.data[i]);
    }

    // Sort candidates
    std::vector<int> sorted_indices(candidate_values.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&candidate_values](const size_t &i, const size_t &j)
              { return candidate_values[i] > candidate_values[j]; });

    // keep only the K bests
    std::vector<std::tuple<int, int>> best_corners_coordinates;
    for (size_t i = 0; i < K && i < candidates_coords.size(); ++i)
    {
        best_corners_coordinates.push_back(candidates_coords[sorted_indices[i]]);
        spdlog::debug("[{}, {}]", std::get<0>(best_corners_coordinates[i]), std::get<1>(best_corners_coordinates[i]));
    }

    return best_corners_coordinates;
}

std::unique_ptr<unsigned char[]> render_harris_cpu(unsigned char *buffer, int width, int height, std::ptrdiff_t, int)
{
    spdlog::debug("Compute grayscale...");
    auto image = grayscale(buffer, width, height);

    spdlog::debug("Compute Harris response...");
    auto harris_res = compute_harris_response(image);

    auto image_mask = image > 0;

    spdlog::debug("Erode shape...");
    auto min_distance = 25;
    auto detect_mask = morph_apply_kernel(image_mask, morph_circle_kernel(min_distance * 2), 0);
    detect_mask = detect_mask * (harris_res > (0.5 * harris_res.max()));

    spdlog::debug("Dilate Harris response...");
    auto dil = morph_apply_kernel(harris_res, morph_circle_kernel(min_distance), 1);
    detect_mask = detect_mask * (harris_res == dil);

    auto best_corners_coordinates = top_k_best_coords_keypoints(detect_mask, harris_res, 10);

    std::ofstream myfile;
    myfile.open("best-keypoints.csv");
    for (int k = 0; k < best_corners_coordinates.size(); ++k)
    {
        myfile << std::get<0>(best_corners_coordinates[k]) << "," << std::get<1>(best_corners_coordinates[k]) << "\n";
        spdlog::debug("[{}, {}]", std::get<0>(best_corners_coordinates[k]), std::get<1>(best_corners_coordinates[k]));
    }

    myfile.close();

    return (detect_mask * 255).to_buffer();
}
