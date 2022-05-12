#pragma once
#include <memory>
#include <cstddef>
#include <vector>

class Matrix {
    public:
        Matrix();
        Matrix(int height, int width);

        Matrix operator*(const Matrix& rhs);
        Matrix operator+(const Matrix& rhs);
        Matrix operator+(const double& rhs);
        Matrix operator-(const Matrix& rhs);
        Matrix operator/(const Matrix& rhs);
        Matrix operator>(const double& rhs);
        Matrix operator==(const Matrix& rhs);

        std::unique_ptr<unsigned char[]> to_buffer();
        double max();
        
        void print();
        void print_size();
        
        std::vector<double> data;
        int height, width;
};
