#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2


# Base directory containing subdirectories with C++ files
BASE_DIR="."

# Iterate over all subdirectories in the base directory
for dir in "$BASE_DIR"/*; do
  if [ -d "$dir" ]; then
    # Navigate into the directory
    cd "$dir" || continue

    # Check if compile.sh exists and is executable
    if [ -f "compile.sh" ] && [ -x "compile.sh" ]; then
      # Run compile.sh and capture output
      OUTPUT=$(./compile.sh 2>&1)
      
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
    cd "$BASE_DIR" || exit
  fi
done
