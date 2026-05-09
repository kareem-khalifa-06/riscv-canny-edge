# Compilers
HOST_CXX := g++
RV_CXX   := riscv64-unknown-linux-gnu-g++


# Flags
HOST_FLAGS := -O2 -std=c++17 -Wall -Wextra
RV_FLAGS   := -march=rv64gcv -O2 -std=c++17 -Wall -Wextra -static

# Dirs
SRC_DIR   := src
RVV_DIR   := rvv
TEST_DIR  := tests
HOST_BUILD := build/host
RV_BUILD   := build/rv

# GoogleTest
GTEST_INC := -I$(HOME)/googletest-install/include
GTEST_LIB := -L$(HOME)/googletest-install/lib -lgtest -lgtest_main -lpthread

# Sources
LIB_SRCS    := $(SRC_DIR)/image_io.cpp $(SRC_DIR)/gaussian.cpp $(SRC_DIR)/sobel.cpp $(SRC_DIR)/magnitude.cpp $(SRC_DIR)/direction.cpp
MAIN_SRC    := $(SRC_DIR)/main.cpp
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.cpp)
RVV_SRCS    := $(wildcard $(RVV_DIR)/*.cpp)

.PHONY: all test canny_rv run clean qemu_test run_qemu_test host run_host

all: canny_rv

test: $(LIB_SRCS) $(TEST_SRCS)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) $^ $(GTEST_LIB) -o $(HOST_BUILD)/host_tests
	./$(HOST_BUILD)/host_tests

canny_rv: $(LIB_SRCS) $(MAIN_SRC) $(RVV_SRCS)
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/canny_rv

VLEN ?= 256
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD)/canny_rv

qemu_test: $(LIB_SRCS) $(SRC_DIR)/main_qemu_test.cpp
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_FLAGS) $^ -o $(RV_BUILD)/qemu_test

run_qemu_test: qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=128 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=256 $(RV_BUILD)/qemu_test
	qemu-riscv64 -cpu rv64,v=true,vlen=512 $(RV_BUILD)/qemu_test

host: $(LIB_SRCS) $(MAIN_SRC)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $^ -o $(HOST_BUILD)/canny_host

run_host: host
	./$(HOST_BUILD)/canny_host $(W) $(H) $(IMG) $(PREFIX)

clean:
	rm -rf build
