#include <iostream>
#include <cstddef>
#include <png.h>

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
