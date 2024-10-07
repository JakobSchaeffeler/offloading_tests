#!/bin/bash

# Base directory containing subdirectories with executables
BASE_DIR="."

# Iterate over all subdirectories in the base directory
for dir in "$BASE_DIR"/*; do
  if [ -d "$dir" ]; then
    # Find all executables starting with 'test_' in the current directory
    for exec in "$dir"/test_*; do
      if [ -f "$exec" ] && [ -x "$exec" ]; then
        # Execute the file and capture the output
        OUTPUT=$("$exec" 2>&1)

        # Check if execution was successful
        if [ $? -ne 0 ]; then
          echo "[FAILED] Execution failed for: $exec"
	  echo "$OUTPUT"
        else
        fi
      fi
    done
  fi
done
