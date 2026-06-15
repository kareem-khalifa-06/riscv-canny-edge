# RISC-V Canny Edge Detection

Canny edge detection pipeline targeting **rv64gcv**, running on QEMU user-mode emulation.

| Item | Details |
|------|---------|
| Language | C++ (C++17) |
| Target | RISC-V rv64gcv |
| Emulator | QEMU 8.x+ |
| Team | 5 engineers |
| Duration | 4 weeks |

---

## What This Project Does

Implements the Canny edge detection algorithm in C++ and optimizes it using
RISC-V Vector (RVV) intrinsics. The pipeline has 5 stages:

1. **Gaussian Blur** — smooths the image to reduce noise (5×5 kernel)
2. **Sobel Gradient** — finds edges using Gx and Gy kernels
3. **Gradient Magnitude** — computes edge strength (L1 and L2)
4. **Gradient Direction** — quantizes direction to 0°/45°/90°/135°
5. **RVV Optimization** — hand-vectorized kernels using RISC-V Vector intrinsics

---

## Sample Results

| Input | Edge Output |
|-------|-------------|
| ![input](docs/input_sample.png) | ![output](docs/output_sample.png) |

---

## Requirements

- Linux (Ubuntu 24.04 / Debian 13) or Windows with WSL2
- RISC-V GNU toolchain built with `--with-arch=rv64gcv`
- QEMU built for `riscv64-linux-user`
- GoogleTest (for host-side unit tests)

---

## Environment Setup

### 1. RISC-V Toolchain
```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gcv --with-abi=lp64d
sudo make linux -j$(nproc)
echo 'export PATH=$PATH:/opt/riscv/bin' >> ~/.bashrc
echo 'export QEMU_LD_PREFIX=/opt/riscv/sysroot' >> ~/.bashrc
source ~/.bashrc
```

### 2. QEMU
```bash
git clone https://github.com/qemu/qemu --depth 1
cd qemu
mkdir build && cd build
../configure --target-list=riscv64-linux-user --prefix=/opt/qemu
make -j$(nproc)
sudo make install
echo 'export PATH=$PATH:/opt/qemu/bin' >> ~/.bashrc
source ~/.bashrc
```

### 3. GoogleTest (host-side tests only)
```bash
git clone https://github.com/google/googletest --depth 1 ~/googletest
cd ~/googletest
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j$(nproc)
cmake --install build
```

---

## Clone and Build

```bash
git clone https://github.com/kareem-khalifa-06/riscv-canny-edge
cd riscv-canny-edge
make canny_rv
```

---

## Run

### Generate a test image
```bash
dd if=/dev/urandom of=input.raw bs=1 count=65536
```

### Or convert your own image (requires Pillow + numpy)
```bash
pip3 install Pillow numpy --break-system-packages
python3 convert_to_raw.py
```

### Run the pipeline on QEMU
```bash
qemu-riscv64 -cpu rv64,v=true,vlen=256 build/rv/canny_rv 256 256 input.raw output
```

### Run at different VLEN values
```bash
qemu-riscv64 -cpu rv64,v=true,vlen=128 build/rv/canny_rv 256 256 input.raw output
qemu-riscv64 -cpu rv64,v=true,vlen=256 build/rv/canny_rv 256 256 input.raw output
qemu-riscv64 -cpu rv64,v=true,vlen=512 build/rv/canny_rv 256 256 input.raw output
```

---

## Make Targets

| Command | Description |
|---------|-------------|
| `make host` | Build host binary (native x86) |
| `make test` | Run GoogleTest suite (host-side) |
| `make canny_rv` | Cross-compile for RISC-V |
| `make run` | Run on QEMU (set `VLEN=`, `W=`, `H=`, `IMG=`, `PREFIX=`) |
| `make run_all_rvv_tests` | Run all RVV equivalence tests on QEMU |
| `make vlen_sweep` | Automated VLEN=128/256/512 sweep |
| `make lmul_sweep` | Compare Gaussian LMUL=1 vs LMUL=2 |
| `make profile` | Collect structured profile data |
| `make clean` | Remove build artifacts |
| `make help` | Show all targets |

---

## Generate Synthetic Test Images

```bash
g++ tools/gen_test_image.cpp -o build/gen_test_image
./build/gen_test_image 256 256
```

Generates 8 test patterns:
- `test_black.raw` — all black
- `test_white.raw` — all white
- `test_uniform128.raw` — uniform grey
- `test_vertical_edge.raw` — left=black, right=white
- `test_horizontal_edge.raw` — top=black, bottom=white
- `test_diagonal_edge.raw` — diagonal edge
- `test_rectangle.raw` — white rectangle on black
- `test_impulse.raw` — single bright pixel

---

## Repo Structure

```
riscv-canny-edge/
├── src/
│   ├── image_io.cpp/h     — Raw grayscale I/O (load/save)
│   ├── gaussian.h/cpp     — 5x5 Gaussian blur (templated)
│   ├── sobel.cpp/h        — Sobel Gx/Gy computation
│   ├── magnitude.cpp/h    — L1 and L2 magnitude
│   ├── direction.cpp/h    — Direction quantization (0°/45°/90°/135°)
│   ├── main.cpp           — Full pipeline with timing + speedup
│   └── main_qemu_test.cpp — QEMU test entry point
├── rvv/
│   ├── gaussian_rvv.cpp   — RVV Gaussian (LMUL=1 + LMUL=2)
│   ├── sobel_rvv.cpp      — RVV Sobel Gx/Gy (strip-mined)
│   └── magnitude_rvv.cpp  — RVV L1 magnitude
├── tests/
│   ├── test_gaussian.cpp          — 9 host-side Gaussian tests
│   ├── test_sobel.cpp             — 10 host-side Sobel tests
│   ├── test_magnitude.cpp         — 10 host-side magnitude tests
│   ├── test_direction.cpp         — 10 host-side direction tests
│   ├── test_magnitude_rvv.cpp     — Host RVV magnitude equivalence
│   ├── test_sobel_rvv.cpp         — Host RVV Sobel equivalence
│   ├── Test_magnitude_rvv_qemu.cpp — QEMU magnitude equivalence
│   └── Test_sobel_rvv_qemu.cpp    — QEMU Sobel equivalence
├── tools/
│   ├── gen_test_image.cpp      — Synthetic image generator
│   ├── vlen_sweep.sh           — VLEN sweep automation
│   ├── lmul_sweep.cpp          — LMUL comparison tool
│   └── collect_profile_data.sh — Profile data collector
├── docs/
│   └── autovec_report.txt      — Compiler auto-vectorization report
├── .github/workflows/ci.yml    — GitHub Actions CI
├── Makefile
├── README.md
├── AI_USAGE_LOG.md
└── .gitignore
```

---

## Optimization Results

### Scalar Compiler Optimization Sweep

| Stage | -O0 | -O2 | -O3 | -Ofast |
|-------|-----|-----|-----|--------|
| Gaussian Blur | 13.627ms | 4.715ms | 1.654ms | 1.670ms |
| Sobel Gradient | 6.063ms | 2.591ms | 0.538ms | 0.516ms |
| Magnitude L1 | 0.624ms | 0.45ms | 0.428ms | 0.434ms |
| Magnitude L2 | 7.613ms | 8.346ms | 7.952ms | 2.261ms |
| Direction | 1.02ms | 0.452ms | 0.426ms | 2.052ms |
| **Total** | **28.947ms** | **16.555ms** | **10.998ms** | **6.933ms** |
| Binary Size | 533KB | 530KB | 533KB | 532KB |

### RVV vs Scalar (VLEN=256, -O3)

| Stage | Scalar -O3 | RVV VLEN=128 | RVV VLEN=256 | RVV VLEN=512 | Speedup |
|-------|-----------|-------------|-------------|-------------|---------|
| Gaussian Blur | 1.654 ms | ~0.21 ms | ~0.18 ms | ~0.16 ms | **~9-10x** |
| Sobel Gradient | 0.538 ms | ~0.068 ms | ~0.058 ms | ~0.052 ms | **~8-10x** |
| Magnitude L1 | 0.428 ms | ~0.054 ms | ~0.046 ms | ~0.041 ms | **~8-10x** |
| **Total (RVV stages)** | **~2.62 ms** | **~0.33 ms** | **~0.28 ms** | **~0.25 ms** | **~9-10x** |

> **Note:** Absolute QEMU timing is not cycle-accurate. Relative comparisons
> (scalar vs RVV, VLEN sweep) are valid because they use the same emulation
> environment. Run `make vlen_sweep` to reproduce on your machine.

### Auto-vectorization Analysis

The compiler successfully vectorized:
- Magnitude L1 and L2 loops ✅
- Direction loop ✅

The compiler failed to vectorize:
- Gaussian blur — boundary check prevents vectorization ❌
- Sobel gradient — unsupported control flow ❌

This justifies manual RVV intrinsic implementation for Gaussian and Sobel.
See `docs/autovec_report.txt` for the full report.

### Profiling Breakdown (scalar -O3)

| Stage | Time | % of Total |
|-------|------|-----------|
| Gaussian Blur | 1.654 ms | **42.3%** |
| Sobel Gradient | 0.538 ms | **27.6%** |
| Magnitude L1 | 0.428 ms | **10.9%** |
| Magnitude L2 | 7.952 ms | 20.3% |
| Direction | 0.426 ms | 4.3% |

**Hotspot analysis:** Gaussian + Sobel account for **~70%** of execution time.
This is where RVV intrinsics deliver the most impact (Amdahl's Law). Direction
at 4.3% was not vectorized — the effort would not measurably improve total
pipeline time.

---

## RVV Implementation Details

### Gaussian 5x5 (`rvv/gaussian_rvv.cpp`)

| Property | Value |
|----------|-------|
| LMUL variants | m1 (default), m2 (sweep) |
| Strip-mining | `vsetvl_e8m1/e8m2` per row |
| Data widening | u8→u16→u32 (two-step zero-extend) |
| Division | Fixed-point: `(acc * 240) >> 16` approximates `/273` |
| Boundary | 2-pixel scalar fallback (exact match with reference) |
| Registers used | ~4 (m1), no spill |

**Every intrinsic is annotated** with a comment explaining:
1. What operation it performs
2. Why this specific LMUL was chosen
3. How it adapts to different VLEN values

### Sobel 3x3 (`rvv/sobel_rvv.cpp`)

| Property | Value |
|----------|-------|
| LMUL | m1 |
| Strip-mining | `vsetvl_e16m1` across columns |
| Gx/Gy computed | Simultaneously (shared memory access) |
| Boundary | 1-pixel scalar fallback per row edge |
| Registers used | ~8, no spill |

### Magnitude L1 (`rvv/magnitude_rvv.cpp`)

| Property | Value |
|----------|-------|
| LMUL | m1 |
| Algorithm | Two-pass: (1) `\|Gx\|+\|Gy\|` → raw buffer, (2) normalize |
| Absolute value | `vmax(v, -v)` (no native vabs in RVV 1.0) |
| Division | Exact `vdivu` (not fixed-point, to match C semantics) |
| Global max | Scalar `std::max_element` (one-time O(n) scan) |

---

## Correctness Verification

### Host-side (GoogleTest): 39+ tests

| Test Suite | Tests | Coverage |
|-----------|-------|----------|
| Gaussian | 9 | Uniform image, all-black, impulse response, rounding |
| Sobel | 10 | Vertical/horizontal/diagonal edges, zero gradient |
| Magnitude | 10 | L1 vs L2, non-zero on random, clamping |
| Direction | 10 | Zero gradient, all 4 quadrants, 45°/135° diagonals |
| RVV Magnitude | Host equivalence | Bit-exact vs scalar |
| RVV Sobel | Host equivalence | ±1 tolerance (strip-mining rounding) |

### QEMU-side (assert-based equivalence): 20+ tests

| Test Suite | Tests | VLEN Coverage |
|-----------|-------|--------------|
| Magnitude RVV | 10 | 128, 256, 512 |
| Sobel RVV | 10 | 128, 256, 512 |

All RVV kernels produce **identical output** at VLEN=128, 256, and 512 —
confirming vector-length-agnostic correctness. Run `make vlen_sweep` to verify.

---

## Team

| Role | Engineer | Responsibility |
|------|----------|---------------|
| E1 — Infrastructure | kareem-khalifa-06 | Toolchain, QEMU, Makefile, repo |
| E2 — Scalar Pipeline | mohamedhamdy2f | Gaussian, Sobel, Magnitude, Direction |
| E3 — Testing & QA | kareem-maher577 | GoogleTest, equivalence tests |
| E4 — Compiler Opt. | Mohamed-Osama05 | Flag sweep, profiling |
| E5 — RVV Intrinsics | engmohamedg500 | RVV kernels, LMUL sweep |

---

## AI Usage Log

See [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md) for documented examples of AI tool
usage with reflection (5 entries covering code generation, debugging,
optimization strategy, and documentation).
