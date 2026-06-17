# AI Usage Log

This log documents how AI tools (Claude by Anthropic) were used during the project.
Each entry follows the format: Question → Suggestion → What we changed → What we learned.

---

## Entry 1 — Compiler Error: cannot open input.raw

**Question asked:**
Why does QEMU give "cannot open input.raw" error even though the file exists?

**What AI suggested:**
The problem was that the Makefile was using `riscv64-unknown-elf-g++` (bare-metal 
compiler) which does not support file I/O through QEMU. AI suggested switching to 
`riscv64-unknown-linux-gnu-g++` (Linux compiler) and setting QEMU_LD_PREFIX.

**What we changed:**
We changed the compiler in the Makefile from `riscv64-unknown-elf-g++` to 
`riscv64-unknown-linux-gnu-g++` and added:
`export QEMU_LD_PREFIX=/opt/riscv/sysroot` to ~/.bashrc

**What we learned:**
The elf toolchain is bare-metal — it has no operating system support.
The linux-gnu toolchain supports Linux system calls like fopen/fread through QEMU.
This is a critical difference in embedded cross-compilation.

---

## Entry 2 — Why Divide by 273 in Gaussian Blur?

**Question asked:**
Why do we divide by 273 specifically in the Gaussian blur? Is that an example?

**What AI suggested:**
273 is the exact sum of all kernel coefficients (2+4+5+4+2+4+9+12+...). 
Dividing by the sum of the kernel weights normalizes the result so brightness 
is preserved. If every pixel = 255, the output should also be 255.

**What we changed:**
Nothing — we verified the math ourselves by adding up the kernel values and 
confirmed 17+38+49+38+17 = 273.

**What we learned:**
The division value is never arbitrary — it is always the sum of kernel weights.
This is called normalization and ensures the blur does not change overall brightness.

---

## Entry 3 — Zero-Padding vs Clamping

**Question asked:**
What is zero-padding and why do we use it instead of clamping?

**What AI suggested:**
Zero-padding means treating out-of-bounds pixels as 0 (black) instead of 
repeating the border pixel. The guide recommends zero-padding because it 
simplifies vectorization later — the boundary check becomes a simple skip.

**What we changed:**
We used zero-padding in both gaussian.cpp and sobel.cpp using:
`if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;`

**What we learned:**
Code structure affects vectorizability. The auto-vectorization report showed
that this boundary check prevents the compiler from auto-vectorizing the 
Gaussian and Sobel loops. This is why manual RVV intrinsics are needed.

---

## Entry 4 — C++ Templates for Convolution

**Question asked:**
How do we make the Gaussian function a C++ template as the guide requires?

**What AI suggested:**
Use three template parameters:
- PixelT = pixel type (uint8_t for grayscale)
- AccumT = accumulator type (int32_t to avoid overflow)
- KernT  = kernel coefficient type (int16_t)

AI also suggested a convenience wrapper so callers don't need to specify types.

**What we changed:**
We verified the types manually:
- uint8_t pixel × int16_t kernel = up to 255×15 = 3825 → needs int32_t
- 25 accumulations × 3825 = 95625 → still fits int32_t ✅
We kept the template but added the inline wrapper for cleaner call sites.

**What we learned:**
Templates in C++ must be in the header file because the compiler needs to see
the full template when generating specialized versions. The three-type design
is a professional pattern used in production vision libraries.

---

## Entry 5 — Auto-vectorization Analysis

**Question asked:**
What does the auto-vectorization report mean and how do we interpret it?

**What AI suggested:**
Run with `-fopt-info-vec-all` and grep for "not vectorized" to see which loops
the compiler rejected. The messages explain why:
- "unsupported control flow" = the if statement inside the loop blocks it
- "not profitable" = compiler decided it's not worth vectorizing

**What we changed:**
We saved the report to docs/autovec_report.txt and counted vector instructions:
`objdump -d binary | grep -c vset` → 14 instructions at -O3

**What we learned:**
The compiler successfully vectorized magnitude and direction but failed on
Gaussian and Sobel because of boundary checks. This data directly justifies
why we need manual RVV intrinsics — the compiler cannot do it automatically.
This is Amdahl's law in practice: optimize where the compiler cannot help.

---

## Entry 6 — Repository Audit and Bug Fixes

**Question asked:**
We asked Claude to review our complete GitHub repository, assess which project
phases were complete, and identify any bugs or issues before the final submission.

**What AI suggested:**
After cloning and reading all 32 source files, Claude identified three bugs:
1. `tests/test_direction_rvv_qemu.cpp` (lowercase) was a byte-for-byte duplicate
   of `tests/Test_direction_rvv_qemu.cpp` (uppercase, which is what the Makefile uses).
2. `tests/test_magnitude_rvv.cpp` and `tests/test_sobel_rvv.cpp` called RVV functions
   (`magnitude_l1_rvv`, `sobel_rvv`) inside GoogleTest without `#ifdef __riscv_v` guards.
   They were not in any Makefile target and could not compile on the host — orphaned files.
3. `bonus_test` was listed in `.PHONY` but had no rule definition, meaning
   `make bonus_test` would crash with "No rule to make target" during a demo.

Claude also added a new `opt_sweep` Makefile target to build at O0/O2/O3/Os/Ofast
and print binary sizes automatically, replacing a manual step from Phase 4.

**What we changed:**
- Deleted the 3 orphaned/duplicate files with `git rm`
- Added the `bonus_test` rule to the Makefile (runs full pipeline with NMS + hysteresis)
- Added the `opt_sweep` rule to the Makefile (Phase 4 compiler sweep)
- Committed as PR #12 and merged to main

**What we learned:**
Always run `git diff --stat` before a demo — a missing Makefile rule in `.PHONY`
is invisible until it crashes in front of the instructor. Static analysis of the
repository (not just the code) catches integration-level bugs that unit tests miss.
The `#ifdef __riscv_v` guard pattern should be used in any file that calls RVV
intrinsics, to prevent accidental host compilation.
