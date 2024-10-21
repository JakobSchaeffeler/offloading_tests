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
  FLAGS=$3
fi


# Base directory containing subdirectories with C++ files
BASE_DIR="tests"
FULL_PATH=$(realpath "$BASE_DIR")

# Iterate over all subdirectories in the base directory
for dir in "$FULL_PATH"/*; do
  if [ -d "$dir" ]; then
    # Navigate into the directory
    cd "$dir" || continue

    # Check if compile.sh exists and is executable
    if [ -f "compile.sh" ] && [ -x "compile.sh" ]; then
      # Run compile.sh and capture output
      echo "Compiling $dir"
      OUTPUT=$(./compile.sh $COMPILER $ARCH "$FLAGS" 2>&1)
      # If compilation fails, print an error and the output
      if [ $? -ne 0 ]; then
        echo "Compilation failed in $dir"
        echo "Error output:"
        echo "$OUTPUT"
      fi
      # If compilation fails, print an error
      if [ $? -ne 0 ]; then
        echo "Compilation failed in directory: $dir"
      fi
    else
      echo "No executable compile.sh found in directory: $dir"
    fi

    # Navigate back to the base directory
    cd "$FULL_PATH" || exit
  fi
done
