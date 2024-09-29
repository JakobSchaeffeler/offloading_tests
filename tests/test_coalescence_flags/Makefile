# Default settings
CC=clang++# Default to clang++, but can be overridden on the command line (e.g., make CC=gcc)
GPU_ARCH =gfx908# Default GPU architecture, can be overridden (e.g., make GPU_ARCH=gfx906)
.PHONY: test_coalesced
CXXFLAGS_COMMON =   # Common flags for all compilers
CXXFLAGS_EXTRA = 

# Add compiler-specific flags
ifeq ($(CC), gcc)
    CXXFLAGS_COMPILER = -std=c++11  # g++ specific flags

else ifeq ($(CC), clang++)
	ifeq ($(findstring gfx, $(GPU_ARCH)),gfx)
    	CXXFLAGS_COMPILER = -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(GPU_ARCH)
	else ifeq ($(findstring sm, $(GPU_ARCH)),sm)
		CXXFLAGS_COMPILER = -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=$(GPU_ARCH)
	else 
		$(error Unsupported architecture: $(GPU_ARCH))
	endif
else	
    $(error Unsupported compiler: $(CC))
endif

test_coalesced: test_coalesced.cpp
		$(CC) $(CXXFLAGS_COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS_EXTRA) $< -o $@


clean:
	rm -f test_threads_default test_threads_explicit test_threads_explicit_with_limit test_threads_explicit_as_const
