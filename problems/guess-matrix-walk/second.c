#include <stdint.h>

void solve(const int32_t* A, int64_t* R, int32_t N) {
    int64_t sum = 0;
    for (int32_t j = 0; j < N; ++j)
        for (int32_t i = 0; i < N; ++i) sum += A[(int64_t)i * N + j];
    R[0] = sum;
}
