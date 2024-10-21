# Default settings
CC=clang++# Default to clang++, but can be overridden on the command line (e.g., make CC=gcc)
GPU_ARCH =gfx908# Default GPU architecture, can be overridden (e.g., make GPU_ARCH=gfx906)

CXXFLAGS_COMMON =   # Common flags for all compilers
CXXFLAGS_EXTRA = 
CXXFLAGS_COMPILER_OMP =

.PHONY: test_cuda test_hip test_omp

test_omp: test_omp.cpp
		$(CC) $(CXXFLAGS_COMPILER_OMP) $(CXXFLAGS_COMMON) $(CXXFLAGS_EXTRA) -O3 $< -o $@

test_cuda: test_cuda.cu
		$(CC) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS_EXTRA) -O3 $< -o $@

test_hip: test_hip.cpp
		$(CC) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS_EXTRA) -O3 $< -o $@


clean:
	rm -f test_cuda test_hip test_omp
