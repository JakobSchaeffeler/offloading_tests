#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)


void thread_team_explicit(double* a, double* b, double* c, double scalar, int array_size, int num_threads, int num_teams)
{
#pragma omp target teams distribute parallel for simd num_threads(num_threads) num_teams(num_teams)
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

void thread_team_explicit_with_limit(double* a, double* b, double* c, double scalar, int array_size, int num_threads, int num_teams, int threadsLimit)
{
#pragma omp target teams distribute parallel for simd num_threads(num_threads) num_teams(num_teams) thread_limit(num_threads)
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}





int main(int argc, char *argv[]) {

   if (argc < 4) {
        printf("Parameters: team_num, thread_num, thread_limit.\n");
        exit(1);
    }
    char *pEnd;
    long numTeams;
    long numThreads;
    long threadsLimit;
    numTeams = strtol(argv[1], &pEnd, 10);
    numThreads = strtol(argv[2], &pEnd, 10);
    threadsLimit = strtol(argv[3], &pEnd, 10);


  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* a2 = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b2 = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c2 = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);

  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE])
#pragma omp target enter data map(alloc: a2[0:SIZE], b2[0:SIZE], c2[0:SIZE])

  // init on device
#pragma omp target teams distribute parallel for simd num_threads(numThreads) num_teams(numTeams)
  for (int i = 0; i < SIZE; i++)
  {
    b[i] = i;
    c[i] = 2*i;
    b2[i] = i;
    c2[i] = 2*i;

  }
  double scal = 1.5;
  thread_team_explicit_with_limit(a, b, c, scal, SIZE, num_threads, num_teams, num_threads);
  thread_team_explicit(a2, b2, c2, scal, SIZE, num_threads, num_teams);

#pragma omp target update from(a[0:SIZE])
#pragma omp target update from(a2[0:SIZE])
  double sum = 0;
  double sum2 = 0;
  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum += a[i];
    sum_wanted += i + (2*i)*scal;
    sum2 += a2[i];
  }
  if (sum != sum_wanted || sum2 != sum_wanted){
    std::cout << "Error in thread exlicit runtime test" << std::endl;
    return -1;
  }

  return 0;
#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE], c[0:SIZE])

}
