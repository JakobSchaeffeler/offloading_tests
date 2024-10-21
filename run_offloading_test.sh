#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <compiler> <architecture> [optional offloading flags]"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2

if [ ! -z "$3" ]; then
  FLAGS=$3
fi


./compile.sh $COMPILER $ARCH "$FLAGS"

./run.sh
echo "Correctness tests done"

./run_profiles.sh $ARCH
