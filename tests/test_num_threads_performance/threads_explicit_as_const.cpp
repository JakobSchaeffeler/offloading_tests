#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)
#define NUM_TEAMS 1
#define NUM_THREADS 1


void triad(double* a, double* b, double* c, double scalar, int array_size)
{
    const int num_teams = NUM_TEAMS;
    const int num_threads = NUM_THREADS;
#pragma omp target teams distribute parallel for simd num_threads(NUM_THREADS) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}





int main(){
  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);

  const int num_teams = NUM_TEAMS;
  const int num_threads = NUM_THREADS;
  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE])

  // init on device
#pragma omp target teams distribute parallel for simd num_threads(NUM_THREADS) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
  for (int i = 0; i < SIZE; i++)
  {
    b[i] = i;
    c[i] = 2*i;
  }
  double scal = 1.5;
  triad(a, b, c, scal, SIZE);

#pragma omp target update from(a[0:SIZE])
  double sum = 0;
  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum += a[i];
    sum_wanted += i + (2*i)*scal;
  }
  if (sum != sum_wanted){
    std::cout << "Error in thread exlicit test" << std::endl;
  }

#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE], c[0:SIZE])

}
