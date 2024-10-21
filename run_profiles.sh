#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <architecture>"
    exit 1
fi

performance_exec=""
ARCH=$1
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


python profiling.py --metrics "#Threads" "#Teams" "Grid Size" --verbose 5 --gpu $GPU tests/test_num_threads_performance/test_threads_default "omp_default" tests/test_num_threads_performance/test_threads_explicit "omp_threads_explicit" tests/test_num_threads_performance/test_threads_explicit_with_limit "omp_threads_explicit_limit" tests/test_num_threads_performance/test_threads_explicit_as_const "omp_threads_explicit_const" --test_name "set_threads_at_compilation"

args=$(cat tests/test_num_threads_runtime/args)
python profiling.py --metrics "#Threads" "#Teams" "Grid Size" --verbose 5 --gpu $GPU tests/test_num_threads_runtime/test_threads_default "omp_default" "tests/test_num_threads_runtime/test_threads_explicit $args" "thread_team_explicit"  "tests/test_num_threads_runtime/test_threads_explicit_with_limit $args" "thread_team_explicit_with_limit" --test_name "set_threads_at_runtime"

python profiling.py --metrics "#Threads" "#Teams" "Grid Size" --verbose 5 --gpu $GPU tests/test_reduction/test_reduction_gpu "reduction_gpu" tests/test_reduction/test_reduction_cpu "reduction_cpu" --test_name "reduction_gpu_cpu_comparison"


# Assign arguments to variables
test_names=( "stencil_1d" "atomic_add" "coalesced_access" "uncoal_access" "multiple_access_not_cached" "register_spill" "uniform_branch" "branch_divergence")

python profiling.py --verbose 5 --gpu $GPU tests/performance_tests/$performance_exec "vec_add" tests/performance_tests/test_omp "vec_add" --test_name "vec_add"
# Base command template
for test_name in "${test_names[@]}"
do
  echo "Running performance test: $test_name"
  python profiling.py --verbose 5 --no_rerun --gpu $GPU tests/performance_tests/$performance_exec "$test_name" tests/performance_tests/test_omp "$test_name" --test_name "$test_name"
done

#set_threads_at_compilation_vs_runtime


