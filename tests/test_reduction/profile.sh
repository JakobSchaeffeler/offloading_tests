#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2


if [[ "$ARCH" == *"gfx"* ]]; then 

    # first compiler reduction gpu and get thread/team config
    make test_reduction_gpu CC=$COMPILER GPU_ARCH=$ARCH

    rocprof ./test_reduction_gpu

    CSV_FILE=results.csv
    SUBSTRING=dot

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

    echo "make test_reduction_cpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA=\"-DNUM_THREADS=$NUM_THREADS -DNUM_TEAMS=$NUM_TEAMS\""


    make test_reduction_cpu CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA="-DNUM_THREADS=$NUM_THREADS -DNUM_TEAMS=$NUM_TEAMS"

    #profile to find performance differences

    omniperf profile -n gpu --  ./test_reduction_gpu

    omniperf analyze -p workloads/gpu/MI100/ --list-stats > kernel_stats.txt
    
    KERNEL_LINE=$(grep -m 1 "dot" "kernel_stats.txt")
    KERNEL_NUM=$(echo "$KERNEL_LINE" | awk -F '│' '{print $2}' | xargs)

    omniperf analyze -p workloads/gpu/MI100/ -k $KERNEL_NUM > reduction_gpu_stats.txt

    omniperf profile -n gpu --  ./test_reduction_cpu

    omniperf analyze -p workloads/gpu/MI100/ --list-stats > kernel_stats.txt
    
    KERNEL_LINE=$(grep -m 1 "dot" "kernel_stats.txt")
    KERNEL_NUM=$(echo "$KERNEL_LINE" | awk -F '│' '{print $2}' | xargs)

    omniperf analyze -p workloads/gpu/MI100/ -k $KERNEL_NUM > reduction_cpu_stats.txt

    # Compare Runtime:
    RUNTIME=$(grep -m 1 "dotEv" "reduction_gpu_stats.txt")
    RUNTIME=$(echo "$KERNEL_LINE" | awk -F '│' '{print $5}' | xargs)

    LDS_ALLOC=$(grep -m 1 "7.1.8" "reduction_gpu_stats.txt")
    LDS_ALLOC=$(echo "$KERNEL_LINE" | awk -F '│' '{print $4}' | xargs)

    LDS_INSTR=$(grep -m 1 "12.2.0" "reduction_gpu_stats.txt")
    LDS_INSTR=$(echo "$KERNEL_LINE" | awk -F '│' '{print $4}' | xargs)


    RUNTIME2=$(grep -m 1 "dotEv" "reduction_cpu_stats.txt")
    RUNTIME2=$(echo "$KERNEL_LINE" | awk -F '│' '{print $5}' | xargs)

    LDS_ALLOC2=$(grep -m 1 "7.1.8" "reduction_cpu_stats.txt")
    LDS_ALLOC2=$(echo "$KERNEL_LINE" | awk -F '│' '{print $4}' | xargs)

    LDS_INSTR2=$(grep -m 1 "12.2.0" "reduction_cpu_stats.txt")
    LDS_INSTR2=$(echo "$KERNEL_LINE" | awk -F '│' '{print $4}' | xargs)


    echo "Runtime Reduction GPU: $RUNTIME"
    echo "Runtime Reduction CPU: $RUNTIME2"

    echo "LDS usage by GPU: $LDS_ALLOC"
    echo "LDS usage by CPU: $LDS_ALLOC2"

    echo "LDS Instruction by GPU: $LDS_INSTR"
    echo "LDS Instruction by CPU: $LDS_INSTR2"
fi