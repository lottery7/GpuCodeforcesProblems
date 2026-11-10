#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // GPU code: blockIdx, blockDim and threadIdx are available only here.
}

// solve runs on the CPU and launches the kernel. A, B and C point to GPU memory.
void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
}
