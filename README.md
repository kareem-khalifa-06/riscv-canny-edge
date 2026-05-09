# RISC-V Canny Edge Detection

Canny edge detection pipeline targeting rv64gcv, running on QEMU.
Language: C++ | Team: 5 engineers | Duration: 4 weeks

## Environment Setup (every engineer)
See the team PDF guide for full instructions (Debian 13 or WSL2).

### GoogleTest (run once)
```bash
git clone https://github.com/google/googletest --depth 1 ~/googletest
cd ~/googletest
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j$(nproc)
cmake --install build
```

## Build & Run
```bash
make canny_rv        # cross-compile for RISC-V
make run             # run on QEMU at VLEN=256
make test            # run GoogleTest suite (host-side)
make run_qemu_test   # run equivalence tests at VLEN 128/256/512
make clean           # remove build artifacts
```

## Generate Test Images
```bash
g++ tools/gen_test_image.cpp -o build/gen_test_image
./build/gen_test_image 512 512
```

## Repo Structure
- `src/`   — scalar C++ pipeline
- `rvv/`   — RVV intrinsic implementations  
- `tests/` — GoogleTest unit tests
- `tools/` — test image generator