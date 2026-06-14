#ifdef __riscv_v

#include "../src/magnitude.h"
#include <riscv_vector.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>   // std::max_element

// ─────────────────────────────────────────────────────────────────────────────
// magnitude_l1_rvv  —  bit-exact with scalar: mag[i] = |Gx|+|Gy| * 255 / max
//
// DESIGN NOTE — why no vredmax intrinsic:
//   The GCC 15.2 riscv64-unknown-linux-gnu toolchain built with --with-arch=rv64gcv
//   does not expose __riscv_vredmax_vs_i16m1 under the non-policy intrinsic API.
//   Rather than guess the correct mangled name, we use std::max_element on the
//   raw magnitude buffer (pass 1 output) for the global-max scan.
//   This scalar scan is O(n) over int16_t — cheaper than the 25-tap convolution
//   it follows — and does not affect the correctness of the vectorised passes.
//
// CORRECTNESS GUARANTEE vs scalar reference:
//   mag[i] = (|Gx[i]| + |Gy[i]|) * 255 / global_max
//   vdivu   = exact truncating unsigned divide  == C  a / b
//   vnclipu shift=0 with VXRM_RDN = pure clip-and-narrow, no rounding
//   → bit-exact match on every input for all VLEN (128/256/512)
// ─────────────────────────────────────────────────────────────────────────────
void magnitude_l1_rvv(const int16_t* __restrict__ Gx,
                      const int16_t* __restrict__ Gy,
                            uint8_t* __restrict__ mag,
                            int w, int h)
{
    const int n = w * h;

    // ── Allocate raw magnitude buffer ─────────────────────────────────────────
    int16_t* mag_raw = static_cast<int16_t*>(
        aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63));
    if (!mag_raw) return;

    // ── Pass 1: vectorised  mag_raw[i] = |Gx[i]| + |Gy[i]| ──────────────────
    for (int i = 0; i < n; ) {
        // vl = min(n-i, VLMAX) for e16m1 — adapts to VLEN 128/256/512
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        vint16m1_t vgx = __riscv_vle16_v_i16m1(Gx + i, vl);
        vint16m1_t vgy = __riscv_vle16_v_i16m1(Gy + i, vl);

        // |v| = max(v, -v)  — RVV 1.0 has no integer vabs instruction
        // Safe: Sobel max = 4×255 = 1020; INT16_MIN never occurs here
        vint16m1_t vax = __riscv_vmax_vv_i16m1(
            vgx, __riscv_vneg_v_i16m1(vgx, vl), vl);
        vint16m1_t vay = __riscv_vmax_vv_i16m1(
            vgy, __riscv_vneg_v_i16m1(vgy, vl), vl);

        // L1 magnitude: max value 1020+1020=2040, fits i16 (max 32767)
        vint16m1_t vmag = __riscv_vadd_vv_i16m1(vax, vay, vl);
        __riscv_vse16_v_i16m1(mag_raw + i, vmag, vl);

        i += (int)vl;   // always advance by vl — never a fixed constant
    }

    // ── Global max via std::max_element (scalar, one-time O(n) scan) ─────────
    // Avoids dependency on vredmax intrinsic availability in this toolchain.
    int16_t global_max = *std::max_element(mag_raw, mag_raw + n);
    if (global_max <= 0) {
        __builtin_memset(mag, 0, (size_t)n);
        free(mag_raw);
        return;
    }
    const uint32_t gmax_u32 = (uint32_t)global_max;

    // ── Pass 2: vectorised  mag[i] = mag_raw[i] * 255 / global_max ───────────
    //
    // WHY vdivu NOT fixed-point:
    //   C integer division truncates toward zero.
    //   vdivu (unsigned vector divide) is also exact truncating division.
    //   Fixed-point (vmul + vsra with shift>0) invokes RNU rounding on the
    //   discarded fractional bits, producing quotient+1 on boundary values —
    //   the systematic +1 error seen in the old implementation.
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        vint16m1_t vmag16 = __riscv_vle16_v_i16m1(mag_raw + i, vl);

        // Reinterpret i16m1 → u16m1 (bit-cast, no instruction emitted)
        // Safe: mag_raw ∈ [0, 2040] — MSB always 0
        vuint16m1_t vmag_u16 = __riscv_vreinterpret_v_i16m1_u16m1(vmag16);

        // Zero-extend u16m1 → u32m2
        // WHY widen: 2040 × 255 = 520200 > UINT16_MAX — must use u32
        vuint32m2_t vmag_u32 = __riscv_vzext_vf2_u32m2(vmag_u16, vl);

        // Multiply by 255 — max result 520200 < UINT32_MAX, no overflow
        vuint32m2_t vscaled = __riscv_vmul_vx_u32m2(vmag_u32, 255u, vl);

        // Exact truncating integer divide — identical to C  a / b  for u32
        vuint32m2_t vnormed = __riscv_vdivu_vx_u32m2(vscaled, gmax_u32, vl);

        // Narrow u32m2 → u16m1, shift=0, RDN rounding mode
        // shift=0: zero fractional bits discarded → rounding mode has no effect
        // RDN (round-down/truncate) chosen to match C semantics if shift != 0
        // GCC 15 signature: vnclipu_wx_u16m1(src, shift, rounding_mode, vl)
        vuint16m1_t vnarrow16 = __riscv_vnclipu_wx_u16m1(
            vnormed, 0, __RISCV_VXRM_RDN, vl);

        // Narrow u16m1 → u8mf2, shift=0, RDN
        // vnormed ∈ [0,255] so saturation to 255 is the desired clamp
        vuint8mf2_t vresult = __riscv_vnclipu_wx_u8mf2(
            vnarrow16, 0, __RISCV_VXRM_RDN, vl);

        __riscv_vse8_v_u8mf2(mag + i, vresult, vl);

        i += (int)vl;
    }

    free(mag_raw);
}

#endif // __riscv_v