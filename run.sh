#!/bin/bash

# Base directory containing subdirectories with executables
BASE_DIR="tests"

# Iterate over all subdirectories in the base directory
for dir in "$BASE_DIR"/*; do
  if [ -d "$dir" ]; then
    # Find all executables starting with 'test_' in the current directory
    OUTPUT_CHECK=""
    first_exec=true
    for exec in "$dir"/test_*; do
      if [ -f "$exec" ] && [ -x "$exec" ]; then
        args=""
	if [[ -f "$dir/args" ]]; then
		args=$(cat $dir/args)
	fi
	
	# Execute the file and capture the output
        OUTPUT=$("$exec $args" 2>&1)
	
	if $first_exec; then
		OUTPUT_CHECK="$OUTPUT"
		first_exec=false
	fi
	
        # Check if execution was successful
        if [ $? -ne 0 ]; then
          echo "[FAILED] Execution failed for: $exec"
	  echo "$OUTPUT"
        else
        fi
	if [ "$OUTPUT" != "$OUTPUT_CHECK" ]; then
            echo "[FAILED] Output mismatch for executables in $dir"
            echo "Current Output: $OUTPUT"
            echo "First Output $OUTPUT_CHECK"
        fi

      fi
    done
  fi
done
