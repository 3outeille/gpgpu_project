#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>

constexpr int kRGBASize = 4;
constexpr int niteration = 10;

void BM_Rendering_cpu(benchmark::State &st)
{
  int width;
  int height;
  // auto input_buffer = read_png("../input/city_scape_8k.png", width, height);
  auto input_buffer = read_png("../input/b003.png", width, height);
  int stride = width * kRGBASize;
  std::vector<unsigned char> data(height * stride);

  for (auto _ : st)
    // render_harris_cpu(data.data(), width, height, stride, niteration);
    render_harris_cpu(reinterpret_cast<unsigned char *>(input_buffer), width, height, stride, niteration);

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State &st)
{
  int width;
  int height;
  auto input_buffer = read_png("../input/b003.png", width, height);
  // auto input_buffer = read_png("../input/b003.png", width, height);

  int stride = width * kRGBASize;

  for (auto _ : st)
    render_harris_gpu(reinterpret_cast<unsigned char *>(input_buffer), width, height, stride, niteration);

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Rendering_gpu)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
