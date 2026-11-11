// Author's solution of matrix-multiplication on riscv-rvv: for each row of A, rows of B are scaled and added to the
// row of C with vector fused multiply-adds, which reads B and writes C sequentially.
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void solve(const float* A, const float* B, float* C, int32_t M, int32_t K, int32_t N) {
    for (int32_t i = 0; i < M; ++i) {
        float* c = C + (size_t)i * N;
        for (size_t j = 0; j < (size_t)N;) {
            size_t vl = __riscv_vsetvl_e32m8((size_t)N - j);
            __riscv_vse32_v_f32m8(c + j, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
            j += vl;
        }
        for (int32_t k = 0; k < K; ++k) {
            float a = A[(size_t)i * K + k];
            const float* b = B + (size_t)k * N;
            for (size_t j = 0; j < (size_t)N;) {
                size_t vl = __riscv_vsetvl_e32m8((size_t)N - j);
                vfloat32m8_t acc = __riscv_vle32_v_f32m8(c + j, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, a, __riscv_vle32_v_f32m8(b + j, vl), vl);
                __riscv_vse32_v_f32m8(c + j, acc, vl);
                j += vl;
            }
        }
    }
}
