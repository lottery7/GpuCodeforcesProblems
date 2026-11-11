// Author's solution of vector-addition on riscv-rvv: 64 floats per vector instruction (LMUL = 8, VLEN = 256).
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void solve(const float* A, const float* B, float* C, int32_t N) {
    for (size_t i = 0, n = (size_t)N; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t a = __riscv_vle32_v_f32m8(A + i, vl);
        vfloat32m8_t b = __riscv_vle32_v_f32m8(B + i, vl);
        __riscv_vse32_v_f32m8(C + i, __riscv_vfadd_vv_f32m8(a, b, vl), vl);
        i += vl;
    }
}
