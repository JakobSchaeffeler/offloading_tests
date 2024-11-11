#!/bin/bash

# Base directory containing subdirectories with executables
BASE_DIR="tests"

echo "Running correctness tests"

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
        OUTPUT=$("./$exec" $args 2>/dev/null)
	# Check if execution was successful
        if [ $? -ne 0 ]; then
          OUTPUTERR=$("./$exec" $args 2>&1)
          echo "[FAILED] Execution failed for: $exec"
	  echo "$OUTPUTERR"
        fi
	OUTPUT=$(echo "$OUTPUT" | tr -dc '0-9')
	if $first_exec; then
		OUTPUT_CHECK="$OUTPUT"
		first_exec=false
		FIRST_NAME=$exec
	fi
	if [ "$OUTPUT" != "$OUTPUT_CHECK" ]; then
            echo "[FAILED] Output mismatch for executables in $dir"
            echo "Current Output of $exec: $OUTPUT"
            echo "Output of $FIRST_NAME: $OUTPUT_CHECK"
        fi

      fi
    done
  fi
done
