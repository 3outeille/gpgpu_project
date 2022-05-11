#pragma once
#include <cstddef>
#include <memory>

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
extern "C"
void render_cpu(char* buffer, int width, int height, std::ptrdiff_t stride, int n_iterations = 100);

std::unique_ptr<unsigned char[]> render_harris_cpu(unsigned char *input_buffer, int input_width, int input_height);

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render(char* buffer, int width, int height, std::ptrdiff_t stride, int n_iterations = 100);
