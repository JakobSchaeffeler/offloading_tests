#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)


double dot(double* a, double* b, double* sums)
{
#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
   {
   sums[omp_get_team_num()] = 0;
#pragma omp distribute parallel for simd reduction(+:sums[omp_get_team_num()]) num_threads(NUM_THREADS)  
  for (int i = 0; i < SIZE; i++){
    sums[omp_get_team_num()] += a[i] * b[i];
  }
  }
#pragma omp target update from(sums[0:NUM_TEAMS])
  double sum = 0.0;
  for (int i = 0; i < NUM_TEAMS; i++){
    sum += sums[i];
  }
  return sum;
   
   /*
#pragma omp target teams distribute parallel for simd map(tofrom: sums[0:255]) num_teams(256) num_threads(512) //schedule(static)
*/
}

int main(){
  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* sums = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*NUM_TEAMS);

  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE], sums[0:NUM_TEAMS])

  // init on device
#pragma omp target teams distribute parallel for simd 
  for (int i = 0; i < SIZE; i++)
  {
    a[i] = i;
    b[i] = 2*i;
  }
  double sum = dot(a, b, sums);

  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum += i * (2*i);
  }
  if (sum != sum_wanted){
    std::cout << "Error in reduction cpu test" << std::endl;
  }

#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE])

}





