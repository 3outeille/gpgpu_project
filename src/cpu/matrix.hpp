#pragma once
#include <memory>
#include <cstddef>
#include <vector>

struct Matrix {
    Matrix();
    Matrix(int height, int width);

    Matrix operator*(const Matrix& rhs) const;
    Matrix operator*(const double& rhs) const;
    Matrix operator+(const Matrix& rhs) const;
    Matrix operator+(const double& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator/(const Matrix& rhs) const;
    Matrix operator>(const double& rhs) const;
    Matrix operator==(const Matrix& rhs) const;

    std::unique_ptr<unsigned char[]> to_buffer();
    double max() const;
    
    void print() const;
    void print_size() const;
    
    int height, width;
    std::vector<double> data;
};
