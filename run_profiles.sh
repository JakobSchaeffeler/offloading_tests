#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <architecture>"
    exit 1
fi

# Assign arguments to variables
ARCH=$1
performance_exec=""
if [[ "$ARCH" == *"sm"* ]]; then
	GPU="nvidia"
	performance_exec="test_cuda"
else 
	if [[ "$ARCH" == *"gfx"* ]]; then
		GPU="amd"
		performance_exec="test_hip"
	else 
		echo "Only gfx and sm architectures supported"
		exit -1
	fi
fi

test_names=("vec_add" "stencil_1d" "atomic_add" "coalesced_access" "uncoal_access" "multiple_access_not_cached" "register_spill" "uniform_branch" "branch_divergence")

# Base command template
for test_name in "${test_names[@]}"
do
  echo "Running performance test: $test_name"
  python profiling.py --gpu $GPU performance_tests/$performance_exec "$test_name" performance_tests/test_omp "$test_name" --test_name "$test_name"
done


python profiling.py --gpu $GPU performance_tests/$performance_exec "$test_name" performance_tests/test_omp "$test_name" --test_name "$test_name"
