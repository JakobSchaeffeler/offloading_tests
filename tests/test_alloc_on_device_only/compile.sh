#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <compiler> <architecture> [optional offloading flags]"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2
FLAGS=""

make test_alloc_on_device_only_O0 CC=$COMPILER GPU_ARCH=$ARCH
make test_alloc_on_device_only_O1 CC=$COMPILER GPU_ARCH=$ARCH
make test_alloc_on_device_only_O2 CC=$COMPILER GPU_ARCH=$ARCH
make test_alloc_on_device_only_O3 CC=$COMPILER GPU_ARCH=$ARCH



