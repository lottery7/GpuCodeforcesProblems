#include <stdint.h>

void solve(const uint32_t* A, const uint32_t* B, uint32_t* R, int32_t N) {
    uint32_t count = 0;
    for (int32_t i = 0; i < N; ++i) count += (uint32_t)__builtin_popcount(A[i] ^ B[i]);
    R[0] = count;
}
