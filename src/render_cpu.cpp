#include "render.hpp"
#include <iostream>
#include <png.h>

void render_cpu(char *buffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
    std::cout << "Hello" << std::endl;

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            for (int k = 0; k < 4; k++)
            {
                buffer[i * stride + j * 4 + k] = 255;
            }
        }
    }
    return;
}