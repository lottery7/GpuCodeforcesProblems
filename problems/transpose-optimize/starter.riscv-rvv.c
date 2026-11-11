#include <stdint.h>

void solve(const float* A, float* B, int32_t N, int32_t M) {
    for (int32_t i = 0; i < N; ++i)
        for (int32_t j = 0; j < M; ++j) B[(int64_t)j * N + i] = A[(int64_t)i * M + j];
}
