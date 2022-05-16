#pragma once

#include <spdlog/spdlog.h>

inline void _abortError(const char *msg, const char *fname, int line)
{
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

template <typename T>
__device__ inline T *eltPtr(T *baseAddress, int col, int row, size_t pitch)
{
    return (T *)((char *)baseAddress + row * pitch + col * sizeof(T));
}

const int KERNEL_SIZE = 7;
const int BLOCK_SIZE = 32;
