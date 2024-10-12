#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)

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
    long num_teams;
    long num_threads;
    long threadsLimit;
    num_teams = strtol(argv[1], &pEnd, 10);
    num_threads = strtol(argv[2], &pEnd, 10);
    threadsLimit = strtol(argv[3], &pEnd, 10);


  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);

  // alloc on device
#pragma omp target enter data map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE])

  // init on device
#pragma omp target teams distribute parallel for simd num_threads(num_threads) num_teams(num_teams)
  for (int i = 0; i < SIZE; i++)
  {
    b[i] = i;
    c[i] = 2*i;
  }
  double scal = 1.5;
  thread_team_explicit_with_limit(a, b, c, scal, SIZE, num_threads, num_teams, num_threads);

#pragma omp target update from(a[0:SIZE])
  double sum = 0;
  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum += a[i];
    sum_wanted += i + (2*i)*scal;
  }
  if (sum != sum_wanted){
    std::cout << "Error in thread exlicit runtime test" << std::endl;
    return -1;
  }

  return 0;
#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE], c[0:SIZE])

}
