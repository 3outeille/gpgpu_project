#include "../render.hpp"
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

MatrixGPU compute_harris_response_gpu(MatrixGPU &image)
{
    int size = 3;
    auto img_x = sobel_filter_gpu(image, size, 1);
    auto img_y = sobel_filter_gpu(image, size, 0);

    auto gauss = gauss_filter_gpu(size);

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

// __global__ void get_candidates_values_gpu_kernel(float *output_values, float *harris_res, float *detect_mask, int width, int height, int *output_counter)
// {
//     int i = threadIdx.y + blockIdx.y * blockDim.y;
//     int j = threadIdx.x + blockIdx.x * blockDim.x;

//     if (i >= height || j >= width)
//         return;

//     if (detect_mask[i * width + j]) {
//         output_values[i * width + j] = harris_res[i * width + j];
//         output_counter[i * width + j] += 1;
//     }

// }

// float *get_candidates_values_gpu(MatrixGPU detect_mask, MatrixGPU harris_res)
// {
//     thrust::device_vector<int> output_counter(detect_mask.width * detect_mask.height);
//     thrust::fill(thrust::device, output_counter.begin(), output_counter.end(), 0);

//     auto output_counter_raw = thrust::raw_pointer_cast(output_counter.data());

//     thrust::device_vector<float> output_values(detect_mask.width * detect_mask.height);
//     auto output_values_raw = thrust::raw_pointer_cast(output_values.data());

//     get_candidates_values_gpu_kernel<<<detect_mask.dimGrid(), detect_mask.dimBlock()>>>(
//         output_values_raw,
//         harris_res.to_device_buffer(),
//         detect_mask.to_device_buffer(),
//         detect_mask.width,
//         detect_mask.height,
//         output_counter_raw
//     );

//     cudaDeviceSynchronize();

//     if (cudaPeekAtLastError())
//         abortError("Computation Error");

//     // std::cout << counter << std::endl;
//     // std::cout << output_counter[0] << std::endl;
//     // return std::vector<float>(output_values_raw, output_values_raw + length);
//     return output_values_raw;
// }

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

    thrust::sort(my_zip, my_zip + values.width * values.height, my_sort_functor());

    std::vector<std::tuple<int, int>> res;

    thrust::host_vector<int> host_index = {index_iterator.begin(), index_iterator.begin() + K};
    thrust::host_vector<float> host_values = {values.data.begin(), values.data.begin() + K};

    for (size_t k = 0; k < K && host_values[k]; k++)
    {
        int i = host_index[k] / values.width;
        int j = host_index[k] % values.width;
        res.push_back(std::make_tuple(i, j));
        spdlog::info("[{}, {}]", std::get<0>(res[k]), std::get<1>(res[k]));
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

    auto image_mask = img_grayscale > 0;

    spdlog::debug("Erode shape gpu ...");
    auto min_distance = 25;
    auto eroded_mask = morph_apply_gpu(image_mask, circle_filter_gpu(min_distance * 2), 0);
    auto thresholded_mask = eroded_mask * (harris_res > (0.5 * harris_res.max()));

    spdlog::debug("Dilate Harris response...");
    auto dil = morph_apply_gpu(harris_res, circle_filter_gpu(min_distance), 1);
    auto detect_mask = thresholded_mask * (harris_res == dil);

    auto best_corners_coordinates = top_k_best_coords_keypoints_gpu(detect_mask, harris_res, 10);

    auto res = detect_mask * 255;
    return res.to_host_buffer();
}
