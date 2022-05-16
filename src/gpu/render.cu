#include "matrix.cuh"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include "kernels/helpers.cuh"
#include "kernels.cuh"
#include <fstream>

MatrixGPU compute_harris_response_gpu(MatrixGPU &image)
{
    auto img_x = sobel_filter_gpu(image, 1);
    auto img_y = sobel_filter_gpu(image, 0);

    auto gauss = gauss_filter_gpu((KERNEL_SIZE - 1) / 2);

    auto I_xx = img_x * img_x;
    auto I_xy = img_x * img_y;
    auto I_yy = img_y * img_y;

    auto W_xx = convolution_2D_gpu(I_xx, gauss);
    auto W_xy = convolution_2D_gpu(I_xy, gauss);
    auto W_yy = convolution_2D_gpu(I_yy, gauss);

    auto W_det = (W_xx * W_yy) - (W_xy * W_xy);
    auto W_trace = W_xx + W_yy;

    return W_det / (W_trace + 1.);
}

struct my_sort_functor
{
    template <typename T1, typename T2>
    __host__ __device__ bool operator()(const T1 &t1, const T2 &t2)
    {
        if (thrust::get<1>(t1) > thrust::get<1>(t2))
            return true;
        if (thrust::get<1>(t1) < thrust::get<1>(t2))
            return false;
        if (thrust::get<0>(t1) > thrust::get<0>(t2))
            return true;
        return false;
    }
};

std::vector<std::tuple<int, int>> top_k_best_coords_keypoints_gpu(MatrixGPU detect_mask, MatrixGPU harris_res, int K = 10)
{
    auto values = detect_mask * harris_res;
    thrust::device_vector<int> index_iterator(values.width * values.height);
    thrust::sequence(index_iterator.begin(), index_iterator.end());

    auto my_zip = thrust::make_zip_iterator(thrust::make_tuple(index_iterator.begin(), values.data.begin()));

    // TODO: fitler zero values;
    thrust::sort(my_zip, my_zip + values.width * values.height, my_sort_functor());

    std::vector<std::tuple<int, int>> res;

    thrust::host_vector<int> host_index = {index_iterator.begin(), index_iterator.begin() + K};
    thrust::host_vector<float> host_values = {values.data.begin(), values.data.begin() + K};

    for (size_t k = 0; k < K && host_values[k]; k++)
    {
        int i = host_index[k] / values.width;
        int j = host_index[k] % values.width;
        res.push_back(std::make_tuple(i, j));
    }

    return res;
}

std::unique_ptr<unsigned char[]> render_harris_gpu(unsigned char *input_buffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
    thrust::host_vector<unsigned char> input_host(input_buffer, input_buffer + (height * width * 4));
    thrust::device_vector<unsigned char> input_device = input_host;

    spdlog::debug("Compute grayscale gpu ...");
    auto img_grayscale = grayscale_gpu(input_device, width, height);

    spdlog::debug("Compute Harris response gpu ...");
    auto harris_res = compute_harris_response_gpu(img_grayscale);

    // auto image_mask = img_grayscale > 0;

    // spdlog::debug("Erode shape gpu ...");
    auto min_distance = 25;
    // auto eroded_mask = morph_apply_gpu(image_mask, circle_filter_gpu(min_distance * 2), 0);
    // auto thresholded_mask = eroded_mask * (harris_res > (0.5 * harris_res.max()));
    auto thresholded_mask = harris_res > (0.5 * harris_res.max());

    spdlog::debug("Dilate Harris response...");
    auto dil = morph_apply_gpu(harris_res, circle_filter_gpu(min_distance), 1);
    auto detect_mask = thresholded_mask * (harris_res == dil);

    auto best_corners_coordinates = top_k_best_coords_keypoints_gpu(detect_mask, harris_res, 2000);

    std::ofstream myfile;
    myfile.open("best-keypoints.csv");
    for (int k = 0; k < best_corners_coordinates.size(); ++k)
    {
        myfile << std::get<0>(best_corners_coordinates[k]) << "," << std::get<1>(best_corners_coordinates[k]) << "\n";
        spdlog::debug("[{}, {}]", std::get<0>(best_corners_coordinates[k]), std::get<1>(best_corners_coordinates[k]));
    }

    myfile.close();

    auto res = detect_mask * 255;

    return res.to_host_buffer();
}
