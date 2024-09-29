#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2


if [[ "$ARCH" == *"gfx"* ]]; then 

    tests=("-O0" "-O1" "-O2" "-O3")

    for test in "${tests[@]}"; do
	    make test_coalesced CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA=$test > /tmp/blubb 2>&1
    	    omniperf profile -n gpu --  ./test_coalesced > /tmp/blubb
	    omniperf analyze -p workloads/gpu/MI100/ --list-stats > kernel_stats.txt
	    
	    KERNEL_LINE=$(grep -m 1 "triad" "kernel_stats.txt")
    	    KERNEL_NUM=$(echo "$KERNEL_LINE" | awk -F '│' '{print $2}' | xargs)

    	    omniperf analyze -p workloads/gpu/MI100/ -k $KERNEL_NUM > stats$test.txt

    	    COAL=$(grep -m 1 "16.1.3" "stats$test.txt")
            COAL=$(echo "$COAL" | awk -F '│' '{print $4}' | xargs)

            COALESCABLE=$(grep -m 1 "15.2.3" "stats$test.txt")
            COALESCABLE=$(echo "$COALESCABLE" | awk -F '│' '{print $4}' | xargs)
	   
	    RUNTIME=$(grep -m 1 "triad" "stats$test.txt")
   	    RUNTIME=$(echo "$RUNTIME" | awk -F '│' '{print $5}' | xargs)
	    
	    SPILL=$(grep -m 1 "15.1.9" "stats$test.txt")
   	    SPILL=$(echo "$SPILL" | awk -F '│' '{print $5}' | xargs)


    	    echo "Coalesced Instructions for $test: $COAL"
            echo "Runtime for $test: $RUNTIME"
	    echo "COALESCABLE for $test: $COALESCABLE"
	    echo "Spill instructions per wave for $test: $SPILL"
    done
fi
