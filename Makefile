 # =============================================================================
# RISC-V Canny Edge Detection — Makefile
# =============================================================================

HOST_CXX    := g++
RV_CXX      := riscv64-unknown-linux-gnu-g++

HOST_FLAGS  := -O2 -std=c++17 -Wall -Wextra
RV_FLAGS    := -march=rv64gcv -O2 -std=c++17 -Wall -Wextra -static

SRC_DIR     := src
RVV_DIR     := rvv
TEST_DIR    := tests
TOOLS_DIR   := tools
HOST_BUILD  := build/host
RV_BUILD    := build/rv

GTEST_INC   := -I$(HOME)/.local/include
GTEST_LIB   := -L$(HOME)/.local/lib -lgtest -lgtest_main -lpthread

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
LIB_SRCS    := $(SRC_DIR)/image_io.cpp \
               $(SRC_DIR)/gaussian.cpp \
               $(SRC_DIR)/sobel.cpp \
               $(SRC_DIR)/magnitude.cpp \
               $(SRC_DIR)/direction.cpp \
               $(SRC_DIR)/nms.cpp \
               $(SRC_DIR)/threshold.cpp

MAIN_SRC    := $(SRC_DIR)/main.cpp
RVV_SRCS    := $(wildcard $(RVV_DIR)/*.cpp)

# ---------------------------------------------------------------------------
# CRITICAL: Host tests must NEVER include RVV or QEMU test files.
# Those call __riscv_v-only functions (sobel_rvv, magnitude_l1_rvv, etc.)
# which are undefined on x86_64. They belong in cross-compiled QEMU targets.
# ---------------------------------------------------------------------------
HOST_TEST_SRCS := $(TEST_DIR)/test_gaussian.cpp \
                   $(TEST_DIR)/test_sobel.cpp \
                   $(TEST_DIR)/test_magnitude.cpp \
                   $(TEST_DIR)/test_direction.cpp \
                   $(TEST_DIR)/test_nms.cpp \
                   $(TEST_DIR)/test_threshold.cpp

# ---------------------------------------------------------------------------
# Runtime params
# ---------------------------------------------------------------------------
VLEN  ?= 256
W     ?= 256
H     ?= 256
IMG   ?= input.raw
PREFIX?= output
LOW   ?= 20
HIGH  ?= 50

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all test canny_rv run clean host run_host qemu_test run_qemu_test qemu_rvv_test run_rvv_tests qemu_sobel_rvv_test run_sobel_rvv_tests qemu_gaussian_rvv_test run_gaussian_rvv_tests qemu_direction_rvv_test run_direction_rvv_tests run_all_rvv_tests vlen_sweep lmul_sweep profile help bonus_test

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
all: canny_rv

# ---------------------------------------------------------------------------
# Host-side tests (x86/ARM — scalar only, no RVV symbols)
# ---------------------------------------------------------------------------
test: $(LIB_SRCS) $(HOST_TEST_SRCS)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) $^ $(GTEST_LIB) -o $(HOST_BUILD)/host_tests
	./$(HOST_BUILD)/host_tests

# ---------------------------------------------------------------------------
# RISC-V cross-compiled binary (scalar + RVV intrinsics)
# ---------------------------------------------------------------------------
canny_rv: $(LIB_SRCS) $(MAIN_SRC) $(RVV_SRCS)
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/canny_rv

# ---------------------------------------------------------------------------
# Run on QEMU
# ---------------------------------------------------------------------------
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD)/canny_rv $(W) $(H) $(IMG) $(PREFIX)

run_bonus: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD)/canny_rv $(W) $(H) $(IMG) $(PREFIX) $(LOW) $(HIGH)

# ---------------------------------------------------------------------------
# Native host binary (no QEMU, no RVV)
# ---------------------------------------------------------------------------
host: $(LIB_SRCS) $(MAIN_SRC)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $^ -o $(HOST_BUILD)/canny_host

run_host: host
	./$(HOST_BUILD)/canny_host $(W) $(H) $(IMG) $(PREFIX)

run_host_bonus: host
	./$(HOST_BUILD)/canny_host $(W) $(H) $(IMG) $(PREFIX) $(LOW) $(HIGH)

# ---------------------------------------------------------------------------
# QEMU-side scalar equivalence test
# ---------------------------------------------------------------------------
qemu_test: $(LIB_SRCS) $(SRC_DIR)/main_qemu_test.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_test

run_qemu_test: qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=128 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=256 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=512 $(RV_BUILD)/qemu_test

# ---------------------------------------------------------------------------
# QEMU-side RVV equivalence tests (cross-compiled, run under QEMU)
# ---------------------------------------------------------------------------
qemu_gaussian_rvv_test: $(LIB_SRCS) $(RVV_SRCS) $(TEST_DIR)/Test_gaussian_rvv_qemu.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_gaussian_rvv_test

run_gaussian_rvv_tests: qemu_gaussian_rvv_test
	@for v in 128 256 512; do \
		echo "=== VLEN=$$v ==="; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v $(RV_BUILD)/qemu_gaussian_rvv_test; \
	done

qemu_rvv_test: $(LIB_SRCS) $(RVV_SRCS) $(TEST_DIR)/Test_magnitude_rvv_qemu.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_rvv_test

run_rvv_tests: qemu_rvv_test
	@for v in 128 256 512; do \
		echo "=== VLEN=$$v ==="; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v $(RV_BUILD)/qemu_rvv_test; \
	done

qemu_sobel_rvv_test: $(LIB_SRCS) $(RVV_SRCS) $(TEST_DIR)/Test_sobel_rvv_qemu.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_sobel_rvv_test

run_sobel_rvv_tests: qemu_sobel_rvv_test
	@for v in 128 256 512; do \
		echo "=== VLEN=$$v ==="; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v $(RV_BUILD)/qemu_sobel_rvv_test; \
	done

qemu_direction_rvv_test: $(LIB_SRCS) $(RVV_SRCS) $(TEST_DIR)/Test_direction_rvv_qemu.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_direction_rvv_test

run_direction_rvv_tests: qemu_direction_rvv_test
	@for v in 128 256 512; do \
		echo "=== VLEN=$$v ==="; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v $(RV_BUILD)/qemu_direction_rvv_test; \
	done

run_all_rvv_tests: run_gaussian_rvv_tests run_rvv_tests run_sobel_rvv_tests run_direction_rvv_tests

# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------
vlen_sweep: canny_rv
	@bash $(TOOLS_DIR)/vlen_sweep.sh

lmul_sweep: $(LIB_SRCS) $(RVV_SRCS) $(TOOLS_DIR)/lmul_sweep.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $(LIB_SRCS) $(RVV_SRCS) $(TOOLS_DIR)/lmul_sweep.cpp -o $(RV_BUILD)/lmul_sweep
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD)/lmul_sweep $(W) $(H) $(IMG)

profile: canny_rv
	@bash $(TOOLS_DIR)/collect_profile_data.sh

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  make test                  Host-side GoogleTest (x86)"
	@echo "  make canny_rv              Cross-compile for RISC-V"
	@echo "  make run                   Run on QEMU (VLEN=$(VLEN))"
	@echo "  make run_bonus             Run with custom thresholds LOW= HIGH="
	@echo "  make host                  Build native x86 binary"
	@echo "  make run_host              Run native x86 binary"
	@echo "  make run_qemu_test         Scalar equivalence on QEMU"
	@echo "  make run_gaussian_rvv_tests  Gaussian RVV on QEMU"
	@echo "  make run_rvv_tests         Magnitude RVV on QEMU"
	@echo "  make run_sobel_rvv_tests   Sobel RVV on QEMU"
	@echo "  make run_direction_rvv_tests Direction RVV on QEMU"
	@echo "  make run_all_rvv_tests     All RVV tests on QEMU"
	@echo "  make vlen_sweep            VLEN=128/256/512 sweep"
	@echo "  make lmul_sweep            LMUL=1 vs LMUL=2 comparison"
	@echo "  make profile               Collect profiling data (JSON)"
	@echo "  make clean                 Remove build artifacts"
	@echo ""
	@echo "Runtime variables:"
	@echo "  VLEN=256           Vector length (128/256/512)"
	@echo "  W=256 H=256        Image dimensions"
	@echo "  IMG=path.raw       Input image"
	@echo "  PREFIX=out         Output prefix"
	@echo "  LOW=20 HIGH=50     Threshold values (for run_bonus)"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf build output_vlen*.raw prof_*.raw
