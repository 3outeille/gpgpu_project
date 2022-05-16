#include <cstddef>
#include <memory>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "render.hpp"

#include <iostream>
#include <png.h>

// Usage: ./mandel
int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;

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

  std::unique_ptr<unsigned char[]> output_buffer;

  // Rendering
  spdlog::debug("Runnging {} mode with (w={},h={},niter={}).", mode, width, height, niter);
  if (mode == "CPU")
  {
    output_buffer = render_harris_cpu(reinterpret_cast<unsigned char *>(input_buffer), input_width, input_height, stride, niter);
  }
  else if (mode == "GPU")
  {
    output_buffer = render_harris_gpu(reinterpret_cast<unsigned char *>(input_buffer), input_width, input_height, stride, niter);
  }

  // Save
  // write_png(reinterpret_cast<std::byte *>(output_buffer.get()), input_width, input_height, input_width * 4, output_filename.c_str());
  // spdlog::debug("Output saved in {}.", output_filename);
}
