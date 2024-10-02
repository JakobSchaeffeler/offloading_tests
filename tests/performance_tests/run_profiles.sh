#!/bin/bash

# List of test names
test_names=("vec_add" "stencil_1d" "atomic_add" "coalesced_access" "uncoalesced_access" "multiple_access_not_cached" "register_spill" "uniform_branch" "branch_divergence")

# Base command template
for test_name in "${test_names[@]}"
do
  echo "Running test: $test_name"
  python ~/master/offloading_tests/profiling.py --gpu nvidia test_cuda "$test_name" test_omp "$test_name" --test_name "$test_name"
done
