# ═══════════════════════════════════════════════════════════════════════════════
# Dual-Target Makefile — Host (x86) + RISC-V (rv64gcv)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Compilers ──
HOST_CXX  = g++
RV_CXX    = riscv64-unknown-elf-g++

# ── Flags ──
CXX_STD   = -std=c++17
WARNINGS  = -Wall -Wextra -Wpedantic

# Host flags
HOST_FLAGS = $(CXX_STD) $(WARNINGS) -O2 -I.
HOST_LIBS  = -lgtest -lgtest_main -pthread

# RISC-V flags
RV_ARCH   = -march=rv64gcv
RV_ABI    = -mabi=lp64d
RV_FLAGS  = $(CXX_STD) $(WARNINGS) $(RV_ARCH) $(RV_ABI) -O3 -static -I.

# ── Directories ──
BUILD_DIR = build
HOST_DIR  = $(BUILD_DIR)/host
RV_DIR    = $(BUILD_DIR)/rv

# ── Source files ──
SRCS = src/image_io.cpp src/gaussian.cpp src/sobel.cpp \
       src/magnitude.cpp src/direction.cpp
MAIN = src/main.cpp
MAIN_QEMU = src/main_qemu_test.cpp

RVV_SRCS = rvv/gaussian_rvv.cpp rvv/sobel_rvv.cpp rvv/magnitude_rvv.cpp

# ── Targets ──
.PHONY: all host canny_rv test clean run \
        run_gaussian_rvv_tests run_sobel_rvv_tests \
        run_magnitude_rvv_tests run_all_rvv_tests \
        vlen_sweep lmul_sweep profile help

# ── Default: build host binary ──
all: host

# ═══════════════════════════════════════════════════════════════════════════════
# HOST BUILDS (native x86, for fast iteration with GoogleTest)
# ═══════════════════════════════════════════════════════════════════════════════

host: $(HOST_DIR)/canny_host

$(HOST_DIR)/canny_host: $(SRCS) $(MAIN) | $(HOST_DIR)
	$(HOST_CXX) $(HOST_FLAGS) $^ -o $@ $(HOST_LIBS)

# ── GoogleTest suite ──
test: $(HOST_DIR)/run_tests
	@echo "=== Running host-side GoogleTest suite ==="
	@$(HOST_DIR)/run_tests

$(HOST_DIR)/run_tests: tests/test_gaussian.cpp tests/test_sobel.cpp \
                       tests/test_magnitude.cpp tests/test_direction.cpp \
                       tests/test_magnitude_rvv.cpp tests/test_sobel_rvv.cpp \
                       $(SRCS) | $(HOST_DIR)
	$(HOST_CXX) $(HOST_FLAGS) -DENABLE_RVV_TESTS \
		tests/test_gaussian.cpp tests/test_sobel.cpp \
		tests/test_magnitude.cpp tests/test_direction.cpp \
		tests/test_magnitude_rvv.cpp tests/test_sobel_rvv.cpp \
		$(SRCS) -o $@ $(HOST_LIBS)

# ═══════════════════════════════════════════════════════════════════════════════
# RISC-V BUILDS (cross-compiled for rv64gcv, runs on QEMU)
# ═══════════════════════════════════════════════════════════════════════════════

canny_rv: $(RV_DIR)/canny_rv

$(RV_DIR)/canny_rv: $(SRCS) $(RVV_SRCS) $(MAIN) | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

$(RV_DIR)/canny_rv_qemu: $(SRCS) $(RVV_SRCS) $(MAIN_QEMU) | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

# ═══════════════════════════════════════════════════════════════════════════════
# QEMU EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

# Run with configurable VLEN (default 256)
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) \
		$(RV_DIR)/canny_rv $(W) $(H) $(IMG) $(PREFIX)

# ═══════════════════════════════════════════════════════════════════════════════
# RVV EQUIVALENCE TESTS (QEMU-side)
# ═══════════════════════════════════════════════════════════════════════════════

run_gaussian_rvv_tests: $(RV_DIR)/Test_gaussian_rvv_qemu
	@echo "=== Gaussian RVV Equivalence Tests (QEMU) ==="
	qemu-riscv64 -cpu rv64,v=true,vlen=$(or $(VLEN),256) $(RV_DIR)/Test_gaussian_rvv_qemu

run_sobel_rvv_tests: $(RV_DIR)/Test_sobel_rvv_qemu
	@echo "=== Sobel RVV Equivalence Tests (QEMU) ==="
	qemu-riscv64 -cpu rv64,v=true,vlen=$(or $(VLEN),256) $(RV_DIR)/Test_sobel_rvv_qemu

run_magnitude_rvv_tests: $(RV_DIR)/Test_magnitude_rvv_qemu
	@echo "=== Magnitude RVV Equivalence Tests (QEMU) ==="
	qemu-riscv64 -cpu rv64,v=true,vlen=$(or $(VLEN),256) $(RV_DIR)/Test_magnitude_rvv_qemu

run_all_rvv_tests: run_gaussian_rvv_tests run_sobel_rvv_tests run_magnitude_rvv_tests
	@echo "=== All RVV equivalence tests passed ==="

# ── QEMU test binaries ──
$(RV_DIR)/Test_gaussian_rvv_qemu: tests/Test_gaussian_rvv_qemu.cpp $(SRCS) $(RVV_SRCS) | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

$(RV_DIR)/Test_sobel_rvv_qemu: tests/Test_sobel_rvv_qemu.cpp $(SRCS) $(RVV_SRCS) | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

$(RV_DIR)/Test_magnitude_rvv_qemu: tests/Test_magnitude_rvv_qemu.cpp $(SRCS) $(RVV_SRCS) | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & SWEEP TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# VLEN sweep: test at VLEN=128, 256, 512
vlen_sweep: canny_rv
	@chmod +x tools/vlen_sweep.sh
	@./tools/vlen_sweep.sh $(or $(W),256) $(or $(H),256) $(or $(IMG),input.raw)

# LMUL sweep: compare m1 vs m2 for Gaussian
lmul_sweep: $(RV_DIR)/lmul_sweep
	@echo "=== LMUL Sweep: Gaussian m1 vs m2 ==="
	qemu-riscv64 -cpu rv64,v=true,vlen=$(or $(VLEN),256) \
		$(RV_DIR)/lmul_sweep $(or $(W),512) $(or $(H),512)

$(RV_DIR)/lmul_sweep: tools/lmul_sweep.cpp rvv/gaussian_rvv.cpp | $(RV_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

# Profile data collection at a specific VLEN
profile: canny_rv
	@chmod +x tools/collect_profile_data.sh
	@./tools/collect_profile_data.sh \
		$(or $(VLEN),256) $(or $(W),256) $(or $(H),256) $(or $(IMG),input.raw)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	rm -rf $(BUILD_DIR)

$(HOST_DIR) $(RV_DIR):
	mkdir -p $@

help:
	@echo "Available targets:"
	@echo "  make host              — Build host binary (native x86)"
	@echo "  make test              — Run GoogleTest suite (host-side)"
	@echo "  make canny_rv          — Cross-compile for RISC-V"
	@echo "  make run               — Run on QEMU (set VLEN=, W=, H=, IMG=, PREFIX=)"
	@echo "  make run_all_rvv_tests — Run all RVV equivalence tests on QEMU"
	@echo "  make vlen_sweep        — Test at VLEN=128, 256, 512"
	@echo "  make lmul_sweep        — Compare Gaussian LMUL=1 vs LMUL=2"
	@echo "  make profile           — Collect structured profile data"
	@echo "  make clean             — Remove build artifacts"
	@echo ""
	@echo "Environment variables for 'make run':"
	@echo "  VLEN=128|256|512   Vector length (default: 256)"
	@echo "  W=NNN              Image width  (default: 256)"
	@echo "  H=NNN              Image height (default: 256)"
	@echo "  IMG=path.raw       Input image  (default: input.raw)"
	@echo "  PREFIX=out         Output prefix"
