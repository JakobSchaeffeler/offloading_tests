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
    make test_reduction_gpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"

    # get team/thread config from default benchmark
    ncu ./test_reduction_gpu > ncu_out.txt

    sed -n '/reduction_gpu/,$p' ncu_out.txt > ncu_out_kernel.txt

    BLOCK_LINE=$(grep -m 1 "Block Size" "ncu_out_kernel.txt")
    BLOCK_SIZE=$(echo $BLOCK_LINE | awk '{print $NF}')
    
    GRID_LINE=$(grep -m 1 "Grid Size" "ncu_out_kernel.txt")
    GRID_SIZE=$(echo $GRID_LINE | awk '{print $NF}')

    BLOCK_SIZE="${BLOCK_SIZE//,/}"
    GRID_SIZE="${GRID_SIZE//,/}"

    # build all other benchmarks with team/thread config
    make test_reduction_cpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS" CXXFLAGS_EXTRA="-DNUM_THREADS=$BLOCK_SIZE -DNUM_TEAMS=$GRID_SIZE" CXXFLAGS_COMMON="-O3"

fi


if [[ "$ARCH" == *"gfx"* ]]; then 

    # first compiler reduction gpu and get thread/team config
    make test_reduction_gpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS"

    rocprof ./test_reduction_gpu

    CSV_FILE=results.csv
    SUBSTRING=reduction_gpu

    read -r GRD WGR <<< $(awk -F',' -v substring="$SUBSTRING" '
        $0 ~ substring { 
            grd = $8;
            wgr = $9;
            print grd, wgr;
            exit;  # Stop after finding the first match
        }
    ' "$CSV_FILE")
    echo $GRD
    echo $WGR
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

    make test_reduction_cpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_COMPILER="$FLAGS" CXXFLAGS_EXTRA="-DNUM_THREADS=$NUM_THREADS -DNUM_TEAMS=$NUM_TEAMS"

fi
