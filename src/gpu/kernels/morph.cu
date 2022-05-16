#include "../kernels.cuh"
#include "helpers.cuh"

__global__ void circle_filter_gpu_kernel(char *output_buffer, int size)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size || j >= size)
        return;

    auto y = static_cast<float>(i) + 0.5;
    auto x = static_cast<float>(j) + 0.5;
    auto radius = static_cast<float>(size) / 2;
    auto distance = sqrt(pow(x - radius, 2) + pow(y - radius, 2));
    output_buffer[i * size + j] = distance < radius;
}

std::tuple<thrust::device_vector<char>, int> circle_filter_gpu(int size)
{
    thrust::device_vector<char> output(size * size);

    char *output_buffer_raw = reinterpret_cast<char *>(thrust::raw_pointer_cast(output.data()));

    auto dimGrid = dim3(32, 32);
    int w = std::ceil((float)size / 32);
    auto dimBlock = dim3(w, w);

    circle_filter_gpu_kernel<<<dimGrid, dimBlock>>>(output_buffer_raw, size);
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    return {output, size};
}

__global__ void morph_apply_gpu_kernel(float *output_buffer, const float *input, int width, int height, size_t pitch, const char *kernel, int kernel_size, int mode)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= height || j >= width)
        return;

    auto kernel_center = kernel_size / 2 - 1;

    float value = mode == 0 ? 1.0 : 0.0;
    for (int k_i = 0; k_i < kernel_size; k_i++)
    {
        auto img_i = i + k_i - kernel_center;
        for (int k_j = 0; k_j < kernel_size; k_j++)
        {
            auto img_j = j + k_j - kernel_center;

            if (kernel[k_i * kernel_size + k_j] == 0.)
                continue;

            float img_value = 0;
            if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width)
                img_value = *eltPtr(input, img_j, img_i, pitch);

            if (mode == 0)
                value = fmin(value, img_value);
            else
                value = fmax(value, img_value);
        }
    }
    output_buffer[i * width + j] = value;
}

MatrixGPU morph_apply_gpu(MatrixGPU &input, const std::tuple<thrust::device_vector<char>, int> &kernel, int mode)
{
    // mode => erode: 0, dilate: 1
    MatrixGPU output(input.height, input.width);

    auto kernel_buffer = thrust::raw_pointer_cast(std::get<0>(kernel).data());

    float *input_pitched;
    size_t pitch;
    cudaMallocPitch(&input_pitched, &pitch, input.width * sizeof(float), input.height);
    cudaMemcpy2D(input_pitched, pitch, input.to_device_buffer(), input.width * sizeof(float), input.width * sizeof(float), input.height, cudaMemcpyDeviceToDevice);

    morph_apply_gpu_kernel<<<output.dimGrid(), output.dimBlock()>>>(output.to_device_buffer(), input_pitched, input.width, input.height, pitch, kernel_buffer, std::get<1>(kernel), mode);
    cudaDeviceSynchronize();

    cudaFree(input_pitched);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    return output;
}

template <int AXIS>
__global__ void morph_dilate_gpu_kernel(float *output_buffer, size_t pitch_out, const float *input, int width, int height, size_t pitch_in, int kernel_size)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= height || j >= width)
        return;

    auto kernel_center = kernel_size / 2 - 1;

    float value = 0.0;

    for (int k_i = 0; k_i < kernel_size; k_i++)
    {
        int img_i;
        int img_j;
        if (AXIS == 0)
        {
            img_i = i + k_i - kernel_center;
            img_j = j;
        }
        else
        {
            img_i = i;
            img_j = j + k_i - kernel_center;
        }

        float img_value = 0;
        if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width)
            img_value = *eltPtr(input, img_j, img_i, pitch_in);

        value = fmax(value, img_value);
    }

    float *output_ptr = eltPtr(output_buffer, j, i, pitch_out);
    *output_ptr = value;
}

MatrixGPU morph_dilate_gpu(MatrixGPU &input, int kernel_size)
{
    float *input_pitched;
    size_t pitch_in;
    cudaMallocPitch(&input_pitched, &pitch_in, input.width * sizeof(float), input.height);
    cudaMemcpy2D(input_pitched, pitch_in, input.to_device_buffer(), input.width * sizeof(float), input.width * sizeof(float), input.height, cudaMemcpyDeviceToDevice);

    float *output_pitched;
    size_t pitch_out;
    cudaMallocPitch(&output_pitched, &pitch_out, input.width * sizeof(float), input.height);

    // Vertical
    morph_dilate_gpu_kernel<0><<<input.dimGrid(), input.dimBlock()>>>(output_pitched, pitch_out, input_pitched, input.width, input.height, pitch_in, kernel_size);
    cudaDeviceSynchronize();

    MatrixGPU output(input.height, input.width);

    // Horizontal
    morph_dilate_gpu_kernel<1><<<output.dimGrid(), output.dimBlock()>>>(output.to_device_buffer(), output.width * sizeof(float), output_pitched, input.width, input.height, pitch_out, kernel_size);
    cudaDeviceSynchronize();

    cudaFree(input_pitched);
    cudaFree(output_pitched);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    return output;
}
