#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2



if [[ "$ARCH" == *"sm"* ]]; then 
    # build default benchmark
    make test_omp CC=$COMPILER GPU_ARCH=$ARCH
    
    make test_cuda CC=nvcc GPU_ARCH=$ARCH
fi


if [[ "$ARCH" == *"gfx"* ]]; then 
    make test_omp CC=$COMPILER GPU_ARCH=$ARCH
    
    make test_hip CC=hipcc GPU_ARCH=$ARCH
fi
