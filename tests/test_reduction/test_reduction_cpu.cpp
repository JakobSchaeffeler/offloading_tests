#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)


double dot(double* a, double* b, double* sums)
{
  const int num_teams = (int) NUM_TEAMS;
  const int num_threads = (int) NUM_THREADS;
#pragma omp target teams num_teams(num_teams) thread_limit(num_threads)
   {
  sums[omp_get_team_num()] = 0;
#pragma omp distribute parallel for simd reduction(+:sums[omp_get_team_num()]) num_threads(num_threads)  
    for (int i = 0; i < SIZE; i++){
      sums[omp_get_team_num()] += a[i] + b[i];
    }
  }
#pragma omp target update from(sums[0:num_teams])
  double sum = 0.0;
  for (int i = 0; i < NUM_TEAMS; i++){
    sum += sums[i];
  }
  return sum;
}

int main(){
  // alloc on host
  const int num_teams = (int) NUM_TEAMS;
  const int num_threads = (int) NUM_THREADS;

  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* sums = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*num_teams);

  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE], sums[0:num_teams])

  // init on device
#pragma omp target teams distribute parallel for simd num_threads(NUM_THREADS) thread_limit(NUM_THREADS) num_teams(num_teams) 
  for (int i = 0; i < SIZE; i++)
  {
    a[i] = i;
    b[i] = i;
  }

  double sum = dot(a, b, sums);

  double sum_wanted = 0;
  
  for (int i = 0; i < SIZE; i++){
    sum_wanted += (double)i + (double)(i);
  }
  
  if (sum != sum_wanted){
    std::cout << "Error in reduction cpu test" << std::endl;
  }

#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE])

}





