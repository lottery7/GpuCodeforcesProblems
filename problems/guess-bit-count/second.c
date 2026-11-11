#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void solve(const uint32_t* A, const uint32_t* B, uint32_t* R, int32_t N) {
    size_t n = (size_t)N;
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vuint32m8_t acc = __riscv_vmv_v_x_u32m8(0, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vuint32m8_t x = __riscv_vxor_vv_u32m8(__riscv_vle32_v_u32m8(A + i, vl), __riscv_vle32_v_u32m8(B + i, vl), vl);
        x = __riscv_vsub_vv_u32m8(x, __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(x, 1, vl), 0x55555555u, vl), vl);
        x = __riscv_vadd_vv_u32m8(__riscv_vand_vx_u32m8(x, 0x33333333u, vl),
                                  __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(x, 2, vl), 0x33333333u, vl), vl);
        x = __riscv_vand_vx_u32m8(__riscv_vadd_vv_u32m8(x, __riscv_vsrl_vx_u32m8(x, 4, vl), vl), 0x0F0F0F0Fu, vl);
        x = __riscv_vsrl_vx_u32m8(__riscv_vmul_vx_u32m8(x, 0x01010101u, vl), 24, vl);
        acc = __riscv_vadd_vv_u32m8_tu(acc, acc, x, vl);
        i += vl;
    }
    uint32_t lanes[32];
    __riscv_vse32_v_u32m8(lanes, acc, vlmax);
    uint32_t count = 0;
    for (size_t i = 0; i < vlmax; ++i) count += lanes[i];
    R[0] = count;
}
