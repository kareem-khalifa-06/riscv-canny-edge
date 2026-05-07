# ── Compiler settings ──────────────────────────────────────────────────────────
HOST_CXX     := g++
RISCV_CXX    := riscv64-unknown-linux-gnu-g++

HOST_FLAGS   := -std=c++17 -Wall -Wextra -O2
RISCV_FLAGS  := -std=c++17 -Wall -Wextra -march=rv64gcv -O2

# ── Directories ────────────────────────────────────────────────────────────────
SRC_DIR      := src
INC_DIR      := include
TEST_DIR     := tests
BUILD_DIR    := build

# ── Targets ────────────────────────────────────────────────────────────────────
HOST_BIN     := $(BUILD_DIR)/canny_host
RISCV_BIN    := $(BUILD_DIR)/canny_riscv
TEST_BIN     := $(BUILD_DIR)/run_tests

# ── Source files ───────────────────────────────────────────────────────────────
SRC_FILES    := $(wildcard $(SRC_DIR)/*.cpp)
TEST_FILES   := $(wildcard $(TEST_DIR)/*.cpp)

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: all host riscv test clean run

all: host riscv

# ── Host build (for GoogleTest and local testing) ──────────────────────────────
host: $(BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) -I$(INC_DIR) $(SRC_FILES) -o $(HOST_BIN)

# ── RISC-V build (for QEMU) ────────────────────────────────────────────────────
riscv: $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_FLAGS) -I$(INC_DIR) $(SRC_FILES) -o $(RISCV_BIN)

# ── Test build (host only, links GoogleTest) ───────────────────────────────────
test: $(BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) -I$(INC_DIR) $(TEST_FILES) $(SRC_FILES) \
		-lgtest -lgtest_main -lpthread -o $(TEST_BIN)
	./$(TEST_BIN)

# ── Run on QEMU ────────────────────────────────────────────────────────────────
run: riscv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 $(RISCV_BIN)

# ── Create build directory ─────────────────────────────────────────────────────
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)