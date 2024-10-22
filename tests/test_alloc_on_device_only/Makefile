# Default settings
CC=clang++# Default to clang++, but can be overridden on the command line (e.g., make CC=gcc)
GPU_ARCH =gfx908# Default GPU architecture, can be overridden (e.g., make GPU_ARCH=gfx906)

CXXFLAGS_COMMON =   # Common flags for all compilers
CXXFLAGS_EXTRA = 
CXX_FLAGS_COMPILER =
.PHONY: test_alloc_on_device_only_O0 test_alloc_on_device_only_O1 test_alloc_on_device_only_O2 test_alloc_on_device_only_O3

test_alloc_on_device_only_O0: alloc_on_device_only.cpp
		$(CC) $(CXXFLAGS_EXTRA) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) -O0 $< -o $@

test_alloc_on_device_only_O1: alloc_on_device_only.cpp
		$(CC) $(CXXFLAGS_EXTRA) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) -O1 $< -o $@

test_alloc_on_device_only_O2: alloc_on_device_only.cpp
		$(CC) $(CXXFLAGS_EXTRA) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) -O2 $< -o $@

test_alloc_on_device_only_O3: alloc_on_device_only.cpp
		$(CC) $(CXXFLAGS_EXTRA) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) -O3 $< -o $@
clean:
	rm -f test_alloc_on_device_only_O0 test_alloc_on_device_only_O1 test_alloc_on_device_only_O2 test_alloc_on_device_only_O3

