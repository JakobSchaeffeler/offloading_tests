#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <compiler> <architecture>"
    exit 1
fi

# Assign arguments to variables
COMPILER=$1
ARCH=$2



if [[ "$ARCH" == *"sm"* ]]; then 
    # build default benchmark
    make test_threads_default CC=$COMPILER GPU_ARCH=$ARCH

    # get team/thread config from default benchmark
    ncu ./test_threads_default > ncu_out.txt

    sed -n '/triad/,$p' ncu_out.txt > ncu_out_kernel.txt

    BLOCK_LINE=$(grep -m 1 "Block Size" "ncu_out_kernel.txt")
    BLOCK_SIZE=$(echo $BLOCK_LINE | awk '{print $NF}')
    
    GRID_LINE=$(grep -m 1 "Grid Size" "ncu_out_kernel.txt")
    GRID_SIZE=$(echo $GRID_LINE | awk '{print $NF}')

    BLOCK_SIZE="${BLOCK_SIZE//,/}"
    GRID_SIZE="${GRID_SIZE//,/}"

    # build all other benchmarks with team/thread config
    make test_threads_all CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA="-DNUM_THREADS=$BLOCK_SIZE -DNUM_TEAMS=$GRID_SIZE"

fi


if [[ "$ARCH" == *"gfx"* ]]; then 

    # first compiler reduction gpu and get thread/team config
    make test_threads_default CC=$COMPILER GPU_ARCH=$ARCH

    rocprof ./test_threads_default > /tmp/blubb

    CSV_FILE=results.csv
    SUBSTRING=triad

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

    echo "make test_threads_all CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA=\"-DNUM_THREADS=$NUM_THREADS -DNUM_TEAMS=$NUM_TEAMS\""


    make test_threads_all CC=$COMPILER GPU_ARCH=$ARCH CXXFLAGS_EXTRA="-DNUM_THREADS=$NUM_THREADS -DNUM_TEAMS=$NUM_TEAMS"

    #profile to find performance differences
  tests=("test_threads_explicit" "test_threads_explicit_with_limit" "test_threads_explicit_as_const")

# CSV file and substring
CSV_FILE="results.csv"
SUBSTRING="triad"

    for test in "${tests[@]}"; do
	    rocprof ./$test > /tmp/blubb

	    # Extract GRD and WGR from the CSV file
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
		echo "Error: Failed to extract grd and wgr values for $test."
		continue  # Skip to the next test if extraction fails
	    fi

	    # Remove any spaces from GRD and WGR
	    GRD2=$(echo $GRD | tr -d '[:space:]')
	    WGR2=$(echo $WGR | tr -d '[:space:]')

	    # Calculate NUM_THREADS and NUM_TEAMS
	    NUM_THREADS2=$WGR2
	    NUM_TEAMS2=$(($GRD2 / $WGR2))
    	    if [ "$NUM_THREADS" -ne "$NUM_THREADS2" ] || [ "$NUM_TEAMS" -ne "$NUM_TEAMS2" ]; then
        	echo "Setting threads explicitly failed in $test, expected NUM_THREADS: $NUM_THREADS, got NUM_THREADS2: $NUM_THREADS2, expected NUM_TEAMS: $NUM_TEAMS, got NUM_TEAMS2: $NUM_TEAMS2"
    	    	echo "Removing $test from performance tests"
		for testl in "${tests[@]}"; do
    			if [[ "$testl" != "$test" ]]; then
        			new_tests+=("$test")
    			fi
			done
	    else
        	echo "Thread setting succeeded in $test"
    	    fi

    done  
    omniperf profile -n gpu --  ./test_threads_default > /tmp/blubb

    omniperf analyze -p workloads/gpu/MI100/ --list-stats > kernel_stats.txt
    
    KERNEL_LINE=$(grep -m 1 "triad" "kernel_stats.txt")
    KERNEL_NUM=$(echo "$KERNEL_LINE" | awk -F '│' '{print $2}' | xargs)

    omniperf analyze -p workloads/gpu/MI100/ -k $KERNEL_NUM > default_stats.txt

    RUNTIME=$(grep -m 1 "triad" "default_stats.txt")
    RUNTIME=$(echo "$RUNTIME" | awk -F '│' '{print $5}' | xargs)
    
    LDS_ALLOC=$(grep -m 1 "7.1.8" "default_stats.txt")
    LDS_ALLOC=$(echo "$LDS_ALLOC" | awk -F '│' '{print $4}' | xargs)

    LDS_INSTR=$(grep -m 1 "12.2.0" "default_stats.txt")
    LDS_INSTR=$(echo "$LDS_INSTR" | awk -F '│' '{print $4}' | xargs)

    echo "Runtime for test_threads_default: $RUNTIME"

    echo "LDS usage for test_threads_default: $LDS_ALLOC"

    echo "LDS Instructions for test_threads_default: $LDS_INSTR"

    for test in "${tests[@]}"; do
	    omniperf profile -n gpu --  ./$test > /tmp/blubb

	    omniperf analyze -p workloads/gpu/MI100/ --list-stats > kernel_stats.txt
	    
	    KERNEL_LINE=$(grep -m 1 "triad" "kernel_stats.txt")
	    KERNEL_NUM=$(echo "$KERNEL_LINE" | awk -F '│' '{print $2}' | xargs)

	    omniperf analyze -p workloads/gpu/MI100/ -k $KERNEL_NUM > test_stats.txt

	    # Compare Runtime:
	    RUNTIME2=$(grep -m 1 "triad" "test_stats.txt")
	    RUNTIME2=$(echo "$RUNTIME2" | awk -F '│' '{print $5}' | xargs)

	    LDS_ALLOC2=$(grep -m 1 "7.1.8" "test_stats.txt")
	    LDS_ALLOC2=$(echo "$LDS_ALLOC2" | awk -F '│' '{print $4}' | xargs)

	    LDS_INSTR2=$(grep -m 1 "12.2.0" "test_stats.txt")
	    LDS_INSTR2=$(echo "$LDS_INSTR2" | awk -F '│' '{print $4}' | xargs)
	    
	    

	    echo "Runtime for $test: $RUNTIME2"

	    echo "LDS usage for $test: $LDS_ALLOC2"

	    echo "LDS Instructions for $test: $LDS_INSTR2"
    done
fi
