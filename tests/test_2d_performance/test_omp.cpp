#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 2048 //8192
#define ALIGNMENT (2*1024*1024)

#define INDEX(x, y, N) ((x) + (y) * (N))

void stencil2d(double *in, double *out, int N) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int y = 1; y < N-1; y++) {
        for (int x = 1; x < N-1; x++) {
            out[INDEX(x, y, N)] = (in[INDEX(x-1, y, N)] +
                                   in[INDEX(x+1, y, N)] +
                                   in[INDEX(x, y-1, N)] +
                                   in[INDEX(x, y+1, N)]) / 4.0f;
        }
    }
}

void mat_mul(double *A, double *B, double *C, int N){
#pragma omp target teams distribute parallel for collapse(2)
	for (int row = 0; row < N; row++) {
	    for (int col = 0; col < N; col++) {
		double sum = 0.0;
		for (int i = 0; i < N; i++) {
		    sum += A[row * N + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	    }
	}
}


int main(){
  // alloc on host
  double* A = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE*SIZE);
  double* B = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE*SIZE);
  double* C = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE*SIZE);

  for (int i = 0; i < SIZE; i++)
  {
	for (int j = 0; j < SIZE; j++){
    	A[i*SIZE + j] = i*SIZE + j;
    	B[i*SIZE + j] = i*SIZE -j;
    	C[i*SIZE + j] = 0;
  	}
  }
  // alloc on device
#pragma omp target enter data map(to: A[0:SIZE*SIZE], B[0:SIZE*SIZE], C[0:SIZE*SIZE])

  for(int i = 0; i < 1; i++){
	  stencil2d(A,B,SIZE);
	  mat_mul(A,B,C, SIZE);
  }

#pragma omp target update from(C[0:SIZE*SIZE])
  printf("c0: %f", C[0]);
  printf("c_last %f\n", C[SIZE*SIZE-1]);

#pragma omp target exit data map(release: A[0:SIZE*SIZE], B[0:SIZE*SIZE], C[0:SIZE*SIZE])

}
