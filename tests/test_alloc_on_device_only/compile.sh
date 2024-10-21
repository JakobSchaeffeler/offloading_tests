#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2

make test_alloc_on_device_only_O0 CC=$COMPILER GPU_ARCH=$ARCH
make test_alloc_on_device_only_O3 CC=$COMPILER GPU_ARCH=$ARCH



