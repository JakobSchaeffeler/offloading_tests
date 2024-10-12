#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)


double reduction_gpu(double* a, double* b)
{
  double sum = 0.0;
#pragma omp target enter data map(to: sum)

  #pragma omp target teams distribute parallel for simd reduction(+:sum)

  for (int i = 0; i < SIZE; i++)
  {
    sum += a[i] + b[i];
  }

#pragma omp target exit data map(from: sum)

  return sum;
}

int main(){
  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);

  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE])
  // init on device
#pragma omp target teams distribute parallel for simd 
  for (int i = 0; i < SIZE; i++)
  {
    a[i] = (double) i;
    b[i] = (double)i;
  }

  double sum = reduction_gpu(a, b);

  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum_wanted += (double)i + (double)i;
  }
  
  if (sum != sum_wanted){
    std::cout << "Error in reduction gpu test" << std::endl;
    return -1;
  }
  return 0;

#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE])

}



