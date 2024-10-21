#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 33554432
#define ALIGNMENT (2*1024*1024)
void register_spill(double *data, int N) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++) {
  	double temp[1000];  // Increase array size
        double additional_vars[1000];

        for (int j = 0; j < 1000; j++) {
            temp[j] = data[i] * j;
            additional_vars[j] = data[i] + j;  // Introduce more temporary variables
        }

        double sum = 0;
        for (int j = 0; j < 1000; j++) {
            sum += temp[j] + additional_vars[j];
        }	
        data[i] = sum;
    }
}

int main(){
  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  for (int i = 0; i < SIZE; i++)
  {
    c[i] = i;
  }
  // alloc on device
#pragma omp target enter data map(to: a[0:SIZE], b[0:SIZE])
#pragma omp target enter data map(to:c[0:SIZE])

  register_spill(c, SIZE);

#pragma omp target update from(c[0:SIZE])
  printf("c0: %f", c[0]);
#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE], c[0:SIZE])

}
