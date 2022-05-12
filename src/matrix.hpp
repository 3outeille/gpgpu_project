#pragma once
#include <memory>
#include <cstddef>

class Matrix {
    public:
        Matrix();
        Matrix(int height, int width);
        ~Matrix();

        Matrix(const Matrix&) = delete;
        Matrix& operator=(Matrix const&) = delete;
        Matrix(Matrix &&) = default;
        Matrix& operator=(Matrix &&) = default;

        Matrix operator*(const Matrix& rhs);
        Matrix operator+(const Matrix& rhs);
        Matrix operator+(const double& rhs);
        Matrix operator-(const Matrix& rhs);
        Matrix operator/(const Matrix& rhs);
        Matrix operator>(const double& rhs);

        std::unique_ptr<unsigned char[]> to_buffer();
        
        void print();
        void print_size();
        
        double *data;
        int height, width;
};
