#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    // GPU code: blockIdx, blockDim and threadIdx are available only here.
}

// solve runs on the CPU and launches the kernel. A, B and C point to GPU memory, row by row.
void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul<<<blocks, threads>>>(A, B, C, M, K, N);
}
