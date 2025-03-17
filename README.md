# profiling.py
Python tool to profile and compare executables. The passed executables will be profiled and key performance metrics are reported to the command line and stores in the results directory in CSV files. 

If accumulate flag is set all metrics will be accumulated across all kernels. Otherwise the kernel containing the passed kernel name will be profiled. If the accumulate flag is set the kernel name will be used to differentiate executables with the same names. 

Example:

```
python profiling.py --gpu amd --accumulate --test_name <name of the test> <executable with args> <kernel name> [<executable with args> <kernel name> ...]
                         
```

# hec.py
Python script using the profiling tool to automatically build and profile HeCBench applications. 

To execute compilers and flags to use for compilation have to be specified. Compilation logs are stored in the results directory. 

For example, this command profiles all benchmarks in HecBench that have variants in OpenMP, SYCL and HIP for the AMD GPU architectue gfx908. The results will be reported to the command line and additionally stored in CSV files. 
```
python hec.py --arch="gfx908" --gpu=amd --omp_compiler=<compiler to use for openmp> --omp_flags=<flags to compile openmp application> --sycl_compiler=<compiler to use for sycl> --sycl_flags=<flags to compile sycl application> --hip_compiler=<compiler to use for hip> --hip_flags=<flags to compile hip application>
```



# Requirements

On AMD GPUs omniperf is used to profile applications, thus a working omniperf profiler is required.

On NVIDIA GPUs ncu and nsys is required. 

# Performance Evaluations on MI100 and V100

The performance metrics collected using this tool on V100 and MI100 GPUs are available at [https://github.com/JakobSchaeffeler/offloading_tests_results](https://github.com/JakobSchaeffeler/offloading_tests_results)

