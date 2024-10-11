#!/bin/bash

test_names=("vec_add" "stencil_1d" "atomic_add" "coalesced_access" "uncoal_access" "multiple_access_not_cached" "register_spill" "uniform_branch" "branch_divergence")


echo "Checking for differences in functionality tests"

python comparison.py --metrics "#Threads" "#Teams" "Grid Size" --verbose 10 results/set_threads_at_compilation.csv
python comparison.py --metrics "#Threads" "#Teams" "Grid Size" --verbose 10 results/set_threads_at_runtime.csv


# Base command template
echo "Checking for differences in performance tests..."
for test_name in "${test_names[@]}"
do
  echo "Running performance test: $test_name"
  python comparison.py --verbose 10 results/$performance_exec.csv 
done



