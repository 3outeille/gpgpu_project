#pragma once
#include <memory>
#include <cstddef>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class MatrixCuDevice {
    public:
        MatrixCuDevice();
        MatrixCuDevice(unsigned char *buffer, int height, int width);
        MatrixCuDevice(int height, int width);

        std::unique_ptr<unsigned char[]> to_buffer();
        thrust::host_vector<double> to_host();
        void print();
        void print_size();
        
        thrust::device_vector<double> data_device;

        int height, width;
};
