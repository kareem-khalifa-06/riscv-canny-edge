# Compilers
HOST_CXX := g++
RV_CXX   := riscv64-unknown-linux-gnu-g++

# Flags
HOST_FLAGS := -O2 -std=c++17 -Wall
RV_FLAGS   := -march=rv64gcv -O2 -std=c++17 -Wall

# Dirs
SRC_DIR   := src
RVV_DIR   := rvv
TEST_DIR  := tests
BUILD_DIR := build

# GoogleTest
GTEST_INC := -I$(HOME)/.local/include
GTEST_LIB := -L$(HOME)/.local/lib -lgtest -lgtest_main -lpthread

# Sources
SCALAR_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
RVV_SRCS    := $(wildcard $(RVV_DIR)/*.cpp)
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.cpp)

.PHONY: all test canny_rv run clean

all: canny_rv

# Host-side unit tests (native g++)
test: $(SCALAR_SRCS) $(TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) $^ $(GTEST_LIB) -o $(BUILD_DIR)/host_tests
	./$(BUILD_DIR)/host_tests

# Cross-compile for RISC-V
canny_rv: $(SCALAR_SRCS) $(RVV_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(BUILD_DIR)/canny_rv

# Run on QEMU
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 $(BUILD_DIR)/canny_rv

clean:
	rm -rf $(BUILD_DIR)
