# ─────────────────────────────────────────────────────────────────────────────
# Compilers
# ─────────────────────────────────────────────────────────────────────────────
HOST_CXX := g++
RV_CXX   := riscv64-unknown-linux-gnu-g++

# ─────────────────────────────────────────────────────────────────────────────
# Flags
# ─────────────────────────────────────────────────────────────────────────────
HOST_FLAGS := -O2 -std=c++17 -Wall -Wextra
RV_FLAGS   := -march=rv64gcv -O2 -std=c++17 -Wall -Wextra -static

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
SRC_DIR    := src
RVV_DIR    := rvv
TEST_DIR   := tests
HOST_BUILD := build/host
RV_BUILD   := build/rv

# ─────────────────────────────────────────────────────────────────────────────
# GoogleTest (host-side only)
# ─────────────────────────────────────────────────────────────────────────────
GTEST_INC := -I$(HOME)/.local/include
GTEST_LIB := -L$(HOME)/.local/lib -lgtest -lgtest_main -lpthread

# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────
# Scalar pipeline (shared by host and RISC-V targets)
LIB_SRCS := $(SRC_DIR)/image_io.cpp \
             $(SRC_DIR)/gaussian.cpp \
             $(SRC_DIR)/sobel.cpp \
             $(SRC_DIR)/magnitude.cpp \
             $(SRC_DIR)/direction.cpp

MAIN_SRC  := $(SRC_DIR)/main.cpp

# RVV intrinsic implementations (cross-compiled only — guarded by #ifdef __riscv_v)
RVV_SRCS  := $(wildcard $(RVV_DIR)/*.cpp)

# Host-side GoogleTest files — EXCLUDES RVV and QEMU test files because those
# call magnitude_l1_rvv() which is undefined on x86 (only exists under __riscv_v)
HOST_TEST_SRCS := $(TEST_DIR)/test_gaussian.cpp \
                  $(TEST_DIR)/test_sobel.cpp \
                  $(TEST_DIR)/test_magnitude.cpp

# ─────────────────────────────────────────────────────────────────────────────
# Runtime parameters (override on command line: make run VLEN=512 W=512 H=512)
# ─────────────────────────────────────────────────────────────────────────────
VLEN   ?= 256
W      ?= 256
H      ?= 256
IMG    ?= input.raw
PREFIX ?= output

# ─────────────────────────────────────────────────────────────────────────────
# Phony targets
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all test canny_rv run clean \
        host run_host \
        qemu_test run_qemu_test \
        qemu_rvv_test run_rvv_tests \
        vlen_sweep

# ─────────────────────────────────────────────────────────────────────────────
# Default
# ─────────────────────────────────────────────────────────────────────────────
all: canny_rv

# ─────────────────────────────────────────────────────────────────────────────
# Host-side GoogleTest suite (x86/ARM — no RVV files, no QEMU test files)
# ─────────────────────────────────────────────────────────────────────────────
test: $(LIB_SRCS) $(HOST_TEST_SRCS)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) $^ $(GTEST_LIB) \
		-o $(HOST_BUILD)/host_tests
	./$(HOST_BUILD)/host_tests

# ─────────────────────────────────────────────────────────────────────────────
# Cross-compiled RISC-V binary (scalar + RVV)
# ─────────────────────────────────────────────────────────────────────────────
canny_rv: $(LIB_SRCS) $(MAIN_SRC) $(RVV_SRCS)
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/canny_rv

# ─────────────────────────────────────────────────────────────────────────────
# Run the RISC-V binary on QEMU
# Usage: make run [VLEN=256] [W=256] [H=256] [IMG=input.raw] [PREFIX=output]
# ─────────────────────────────────────────────────────────────────────────────
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD)/canny_rv \
		$(W) $(H) $(IMG) $(PREFIX)

# ─────────────────────────────────────────────────────────────────────────────
# Host-side binary (native x86, no QEMU — useful for quick correctness checks)
# Usage: make run_host [W=256] [H=256] [IMG=input.raw] [PREFIX=output]
# ─────────────────────────────────────────────────────────────────────────────
host: $(LIB_SRCS) $(MAIN_SRC)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $^ -o $(HOST_BUILD)/canny_host

run_host: host
	./$(HOST_BUILD)/canny_host $(W) $(H) $(IMG) $(PREFIX)

# ─────────────────────────────────────────────────────────────────────────────
# QEMU-side scalar equivalence test (assert-based, cross-compiled)
# ─────────────────────────────────────────────────────────────────────────────
qemu_test: $(LIB_SRCS) $(SRC_DIR)/main_qemu_test.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_test

run_qemu_test: qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=128 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=256 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=512 $(RV_BUILD)/qemu_test

# ─────────────────────────────────────────────────────────────────────────────
# QEMU-side RVV equivalence test (scalar vs RVV pixel-by-pixel, cross-compiled)
# ─────────────────────────────────────────────────────────────────────────────
qemu_rvv_test: $(LIB_SRCS) $(RVV_SRCS) $(TEST_DIR)/Test_magnitude_rvv_qemu.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_rvv_test

run_rvv_tests: qemu_rvv_test
	@for VLEN in 128 256 512; do \
		echo "=== VLEN=$$VLEN ==="; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$VLEN $(RV_BUILD)/qemu_rvv_test; \
	done

# ─────────────────────────────────────────────────────────────────────────────
# VLEN sweep — run at 128/256/512 and diff outputs to verify VLA correctness
# Usage: make vlen_sweep W=256 H=256 IMG=input.raw
# ─────────────────────────────────────────────────────────────────────────────
vlen_sweep: canny_rv
	@echo "=== VLEN Sweep ==="
	@for VLEN in 128 256 512; do \
		echo "--- VLEN=$$VLEN ---"; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$VLEN $(RV_BUILD)/canny_rv \
			$(W) $(H) $(IMG) output_vlen$$VLEN; \
	done
	@diff output_vlen128_mag_l1.raw output_vlen256_mag_l1.raw \
		&& echo "128 vs 256: MATCH" || echo "128 vs 256: MISMATCH"
	@diff output_vlen128_mag_l1.raw output_vlen512_mag_l1.raw \
		&& echo "128 vs 512: MATCH" || echo "128 vs 512: MISMATCH"

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf build output_vlen*.raw