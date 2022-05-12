#pragma once
#include <memory>
#include <cstddef>
#include <thrust/host_vector.h>

class MatrixCu {
    public:
        MatrixCu();
        MatrixCu(int height, int width);
        // ~MatrixCu();

        std::unique_ptr<unsigned char[]> to_buffer();
        
        void print();
        void print_size();
        
        thrust::host_vector<double> data;
        int height, width;
};
