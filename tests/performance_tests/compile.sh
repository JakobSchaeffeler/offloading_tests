#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <compiler> <architecture> [optional offloading flags]"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2
FLAGS=""

if [ ! -z "$3" ]; then
  FLAGS="$3"
fi



if [[ "$ARCH" == *"sm"* ]]; then 
    # build default benchmark
    make test_omp CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER_OMP=$FLAGS
    
    make test_cuda CC=nvcc GPU_ARCH=$ARCH
fi


if [[ "$ARCH" == *"gfx"* ]]; then 
    make test_omp CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER_OMP=$FLAGS
    
    make test_hip CC=hipcc GPU_ARCH=$ARCH
fi
