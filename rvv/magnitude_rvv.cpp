#ifdef __riscv_v

#include "../src/magnitude.h"
#include <riscv_vector.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// magnitude_l1_rvv
//
// Computes normalised L1 gradient magnitude using RVV intrinsics.
//
// Algorithm (two-pass, matches scalar reference exactly):
//   Pass 1: mag_raw[i] = |Gx[i]| + |Gy[i]|,  find global_max via vredmax
//   Pass 2: mag[i]     = (mag_raw[i] * 255) / global_max   (normalise to u8)
//
// WHY two passes?
//   Single-pass normalisation would require knowing global_max before writing
//   output — impossible in a forward scan.  Two passes are standard in vision
//   pipelines.  Pass 1 is memory-bandwidth-bound (read Gx,Gy); pass 2 is
//   arithmetic-bound (multiply + divide).  Both vectorise cleanly.
//
// Register budget (LMUL=1, e16):
//   VLEN=128: 8 i16 elements per register; 32 logical regs.
//   We need: vgx, vgy, vax, vay, vmag, vmax_acc = 6 regs.  Comfortable.
// ─────────────────────────────────────────────────────────────────────────────
void magnitude_l1_rvv(const int16_t* __restrict__ Gx,
                      const int16_t* __restrict__ Gy,
                            uint8_t* __restrict__ mag,
                            int w, int h)
{
    const int n = w * h;

    // ── Allocate temporary buffer for raw (un-normalised) magnitudes
    //    We need int16_t because max raw value = 4 × 255 = 1020 > 255.
    int16_t* mag_raw = static_cast<int16_t*>(
        aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63));
    if (!mag_raw) return;   // OOM guard

    // ─────────────────────────────────────────────────────────────────────────
    // PASS 1 — compute |Gx| + |Gy| and find global maximum via vector reduction
    // ─────────────────────────────────────────────────────────────────────────

    // Reduction accumulator: a 1-element vector register holding the running max.
    // Initialised to 0 (minimum possible magnitude).
    //
    // INTRINSIC: __riscv_vmv_v_x_i16m1(0, 1)
    // WHAT:  Create a 1-element i16m1 vector containing 0.
    //        Used as the initial value for vredmax reduction.
    // WHY 1 element: vredmax_vs writes its result into element [0] of the
    //        destination.  We only need 1 element to hold the scalar result.
    // IF VLEN CHANGES: vl=1 is always valid regardless of VLEN.
    vint16m1_t vmax_acc = __riscv_vmv_v_x_i16m1(0, 1);

    for (int i = 0; i < n; ) {
        // INTRINSIC: __riscv_vsetvl_e16m1(n - i)
        // WHAT:  Set vl = min(n-i, VLMAX) for element width e16, LMUL=1.
        // WHY e16m1: Gx/Gy are int16_t.  LMUL=1 gives 32 logical regs —
        //        enough for vgx, vgy, vax, vay, vmag, vmax_acc with headroom.
        // IF VLEN CHANGES: At VLEN=128, vl=8; VLEN=512, vl=32.  No code change.
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        // INTRINSIC: __riscv_vle16_v_i16m1(ptr, vl)
        // WHAT:  Unit-stride load of vl int16 elements.
        // WHY unit-stride: Gx and Gy are SoA (separate arrays), so all elements
        //        in a strip are contiguous — no gather needed.
        vint16m1_t vgx = __riscv_vle16_v_i16m1(Gx + i, vl);
        vint16m1_t vgy = __riscv_vle16_v_i16m1(Gy + i, vl);

        // Absolute value:  vabs = max(v, -v)
        // RVV has no dedicated vabs instruction for integers.
        // The standard idiom is: negate the vector, then take element-wise max.
        //
        // INTRINSIC: __riscv_vneg_v_i16m1(v, vl)
        // WHAT:  Negate every element: result[i] = -v[i].
        // WHY: Combined with vmax, gives |v[i]| = max(v[i], -v[i]).
        //
        // INTRINSIC: __riscv_vmax_vv_i16m1(a, b, vl)
        // WHAT:  Element-wise signed maximum.
        // WHY not vabs: No vabs_v exists in RVV 1.0 integer ISA.
        vint16m1_t vax = __riscv_vmax_vv_i16m1(vgx, __riscv_vneg_v_i16m1(vgx, vl), vl);
        vint16m1_t vay = __riscv_vmax_vv_i16m1(vgy, __riscv_vneg_v_i16m1(vgy, vl), vl);

        // L1 magnitude: |Gx| + |Gy|
        //
        // INTRINSIC: __riscv_vadd_vv_i16m1(a, b, vl)
        // WHAT:  Element-wise addition.
        //        Max value: 1020 + 1020 = 2040 — fits i16 (max 32767). ✓
        // WHY i16 (not i32): Avoids widen; 2040 fits i16 so no overflow.
        vint16m1_t vmag = __riscv_vadd_vv_i16m1(vax, vay, vl);

        // Store raw magnitude for pass 2
        //
        // INTRINSIC: __riscv_vse16_v_i16m1(ptr, v, vl)
        // WHAT:  Unit-stride store of vl int16 elements.
        __riscv_vse16_v_i16m1(mag_raw + i, vmag, vl);

        // Reduction: find maximum in this chunk, accumulate into vmax_acc
        //
        // INTRINSIC: __riscv_vredmax_vs_i16m1(vs2, vs1, vl)
        // WHAT:  Signed horizontal maximum reduction.
        //        result[0] = max(vs1[0], vs2[0], vs2[1], ..., vs2[vl-1])
        //        where vs1 is the scalar (neutral element) register.
        // WHY vs (scalar-init variant): allows chaining across strips —
        //        pass vmax_acc as vs1 so each call extends the running max.
        // IF VLEN CHANGES: vl already captures the correct element count;
        //        the reduction adapts automatically.
        vmax_acc = __riscv_vredmax_vs_i16m1(vmag, vmax_acc, vl);

        i += (int)vl;   // advance by EXACTLY vl — never a fixed stride
    }

    // Extract scalar maximum from element [0] of the reduction result
    //
    // INTRINSIC: __riscv_vmv_x_s_i16m1_i16(v)
    // WHAT:  Move element [0] of a vector register to a scalar integer register.
    // WHY: vredmax result lives in a vector register; we need it as a C int16_t
    //      to use in the scalar normalisation divisor below.
    int16_t global_max = __riscv_vmv_x_s_i16m1_i16(vmax_acc);
    if (global_max <= 0) global_max = 1;    // guard against all-zero input

    // ─────────────────────────────────────────────────────────────────────────
    // PASS 2 — normalise raw magnitudes to [0, 255]
    //
    //   mag[i] = (mag_raw[i] * 255) / global_max
    //
    // Implementation strategy:
    //   Widen i16 → i32, multiply by 255, divide by global_max via
    //   fixed-point (same technique as Gaussian), narrow i32 → u8.
    //
    // Fixed-point: precompute  fp_scale = round(255 * 2^16 / global_max)
    //   Then: normalised = (raw * fp_scale) >> 16
    //   This avoids a vector integer divide (which doesn't exist in RVV 1.0).
    // ─────────────────────────────────────────────────────────────────────────
    const int32_t fp_scale = (int32_t)(((int64_t)255 << 16) / global_max);

    for (int i = 0; i < n; ) {
        // INTRINSIC: __riscv_vsetvl_e16m1
        // WHAT/WHY/IF VLEN CHANGES: same as pass 1.
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        // Load raw i16 magnitudes computed in pass 1
        vint16m1_t vmag = __riscv_vle16_v_i16m1(mag_raw + i, vl);

        // Widen i16 → i32 for multiply (avoid 16-bit overflow)
        //
        // INTRINSIC: __riscv_vsext_vf2_i32m2(v, vl)
        // WHAT:  Sign-extend each i16 element to i32.  Result is LMUL=2.
        // WHY sign-extend (not zero): mag_raw values are signed i16;
        //        though they are always ≥ 0 in practice, sign-extend is
        //        semantically correct and safe.
        vint32m2_t vmag32 = __riscv_vsext_vf2_i32m2(vmag, vl);

        // Multiply by fixed-point scale factor
        //
        // INTRINSIC: __riscv_vmul_vx_i32m2(v, scalar, vl)
        // WHAT:  Multiply every i32 element by the scalar fp_scale.
        //        Max: 2040 * (255 * 65536 / 1) ≈ 34M — fits i32 when
        //        global_max > 0 (guarded above).
        vint32m2_t vscaled = __riscv_vmul_vx_i32m2(vmag32, fp_scale, vl);

        // Arithmetic right shift by 16 (complete the fixed-point divide)
        //
        // INTRINSIC: __riscv_vsra_vx_i32m2(v, 16, vl)
        // WHAT:  Arithmetic right shift by 16 bits (sign-preserving).
        //        Result is the normalised value in [0, 255].
        vint32m2_t vnorm = __riscv_vsra_vx_i32m2(vscaled, 16, vl);

        // Narrow i32m2 → u8m1 in two saturating steps
        //
        // Step A: i32m2 → u16m1 (saturate to [0, 65535])
        // INTRINSIC: __riscv_vnclipu_wx_u16m1(v, 0, vl)
        // WHAT:  Unsigned saturating narrowing: clip each i32 to [0, 65535],
        //        then truncate to 16 bits.  Shift=0 means no additional shift.
        // WHY vnclipu (unsigned): normalised values are ≥ 0; unsigned saturation
        //        correctly clamps negative residuals (rounding artefacts) to 0.
        vuint16m1_t vnarrow16 = __riscv_vnclipu_wx_u16m1(
            __riscv_vreinterpret_v_i32m2_u32m2(vnorm), 0, vl);

        // Step B: u16m1 → u8mf2 (saturate to [0, 255])
        // INTRINSIC: __riscv_vnclipu_wx_u8mf2(v, 0, vl)
        // WHAT:  Unsigned saturating narrowing u16→u8.
        // WHY two steps: RVV vnclipu only halves element width per call.
        // IF VLEN CHANGES: mf2 (fractional LMUL) adapts element count correctly.
        vuint8mf2_t vresult = __riscv_vnclipu_wx_u8mf2(vnarrow16, 0, vl);

        // Store final u8 output
        //
        // INTRINSIC: __riscv_vse8_v_u8mf2(ptr, v, vl)
        // WHAT:  Unit-stride store of vl bytes.
        __riscv_vse8_v_u8mf2(mag + i, vresult, vl);

        i += (int)vl;
    }

    free(mag_raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public C-linkage wrapper (called from main.cpp / equivalence tests)
// Declared in magnitude.h as magnitude_l1_rvv()
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: If you want to call this from main.cpp, add to magnitude.h:
//   void magnitude_l1_rvv(const int16_t* Gx, const int16_t* Gy,
//                          uint8_t* mag, int w, int h);

#endif // __riscv_v