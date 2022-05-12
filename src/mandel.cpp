#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "render.hpp"

#include <iostream>

std::byte *read_png(const char *file_name, int &width, int &height)
{
  char header[8]; // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(file_name, "rb");
  fread(header, 1, 8, fp);

  /* initialize stuff */
  auto png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  auto info_ptr = png_create_info_struct(png_ptr);

  setjmp(png_jmpbuf(png_ptr));

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  // color_type = png_get_color_type(png_ptr, info_ptr);
  // bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  setjmp(png_jmpbuf(png_ptr));

  std::cout << height << 'x' << width << " | " << png_get_rowbytes(png_ptr, info_ptr) << std::endl;

  auto row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++)
  {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);

  fclose(fp);

  auto res = (std::byte *)malloc(width * height * 4 * sizeof(std::byte));

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        res[i * (width * 4) + j * 4 + k] = static_cast<std::byte>(row_pointers[i][j * 3 + k]);
      }

      res[(width * i + j) * 4 + 3] = static_cast<std::byte>(255);
    }
  }

  return res;
}

void write_png(const std::byte *buffer,
               int width,
               int height,
               int stride,
               const char *output_filename)
{
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE *fp = fopen(output_filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}

// Usage: ./mandel
int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  // int input_height;
  // int input_width;
  // auto input_img = read_png("../img/b005.png", input_width, input_height);
  // write_png(input_img, input_width, input_height, input_width * 4, "test_save.png");

  std::string output_filename = "output.png";
  std::string input_filename = "input.png";
  std::string mode = "GPU";
  int width = 4800;
  int height = 3200;
  int niter = 100;

  CLI::App app{"mandel"};
  app.add_option("-o", output_filename, "Output image");
  app.add_option("-i", input_filename, "Output image");
  app.add_option("niter", niter, "number of iteration");
  app.add_option("width", width, "width of the output image");
  app.add_option("height", height, "height of the output image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);

  int input_width;
  int input_height;
  auto input_buffer = read_png(input_filename.c_str(), input_width, input_height);

  // Create buffer
  constexpr int kRGBASize = 4;
  int stride = width * kRGBASize;
  // auto output_buffer = std::make_unique<std::byte[]>(height * stride);

  std::unique_ptr<unsigned char[]> output_buffer;

  // Rendering
  spdlog::info("Runnging {} mode with (w={},h={},niter={}).", mode, width, height, niter);
  if (mode == "CPU")
  {
    // render_cpu(reinterpret_cast<char *>(output_buffer.get()), width, height, stride, niter);
    output_buffer = render_harris_cpu(reinterpret_cast<unsigned char *>(input_buffer), input_width, input_height);
  }
  else if (mode == "GPU")
  {
    render(reinterpret_cast<char *>(output_buffer.get()), width, height, stride, niter);
  }

  // Save
  write_png(reinterpret_cast<std::byte *>(output_buffer.get()), input_width, input_height, input_width * 4, output_filename.c_str());
  spdlog::info("Output saved in {}.", output_filename);
}
