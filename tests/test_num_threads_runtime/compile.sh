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


if [[ "$ARCH" == *"sm"* ]]; then 
    # build default benchmark
    make test_threads_default CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"

    make test_threads_explicit CC=$COMPILER GPU_ARCH=$ARCH  CXXFLAGS_COMPILER="$FLAGS"
    make test_threads_explicit_with_limit CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"
    
    ncu ./test_threads_default > ncu_out.txt

    sed -n '/omp_default/,$p' ncu_out.txt > ncu_out_kernel.txt

    BLOCK_LINE=$(grep -m 1 "Block Size" "ncu_out_kernel.txt")
    BLOCK_SIZE=$(echo $BLOCK_LINE | awk '{print $NF}')
    
    GRID_LINE=$(grep -m 1 "Grid Size" "ncu_out_kernel.txt")
    GRID_SIZE=$(echo $GRID_LINE | awk '{print $NF}')

    BLOCK_SIZE="${BLOCK_SIZE//,/}"
    GRID_SIZE="${GRID_SIZE//,/}"
    echo "$GRID_SIZE $BLOCK_SIZE $BLOCK_SIZE" > args
fi


if [[ "$ARCH" == *"gfx"* ]]; then 

    # first compiler reduction gpu and get thread/team config
    make test_threads_default CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"

    rocprof ./test_threads_default > /tmp/blubb

    CSV_FILE=results.csv
    SUBSTRING=omp_default

    read -r GRD WGR <<< $(awk -F',' -v substring="$SUBSTRING" '
        $0 ~ substring { 
            grd = $8;
            wgr = $9;
            print grd, wgr;
            exit;  # Stop after finding the first match
        }
    ' "$CSV_FILE")

    # Check if the values were successfully extracted
    if [ -z "$GRD" ] || [ -z "$WGR" ]; then
        echo "Error: Failed to extract grd and wgr values."
        exit 1
    fi

    GRD=$(echo $GRD | tr -d '[:space:]')
    WGR=$(echo $WGR | tr -d '[:space:]')

    NUM_THREADS=$WGR
    NUM_TEAMS=$(($GRD / $WGR))
    
    #build reduction on cpu with same config
    echo "$NUM_TEAMS $NUM_THREADS $NUM_THREADS" > args
    make test_threads_explicit_with_limit CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"
    make test_threads_explicit CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"
fi
