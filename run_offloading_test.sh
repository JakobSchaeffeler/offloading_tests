#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2


./compile.sh $COMPILER $ARCH

./run.sh

./run_profiles.sh $ARCH
