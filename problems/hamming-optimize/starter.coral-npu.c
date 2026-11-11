#include <stdint.h>

void solve(const uint32_t* A, const uint32_t* B, uint32_t* R, int32_t N) {
    uint32_t count = 0;
    for (int32_t i = 0; i < N; ++i) {
        // one differing bit at a time
        for (uint32_t diff = A[i] ^ B[i]; diff != 0; diff >>= 1) count += diff & 1;
    }
    R[0] = count;
}
