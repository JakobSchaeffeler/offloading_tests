#include <omp.h>
#include <cstdlib> 
#include <iostream>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)

void vec_add(double *a, double *b, double *c){
#pragma omp target teams distribute parallel for
	for (int i = 0; i < SIZE; i++) {
    		c[i] = a[i] + b[i];
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

void stencil_1d(double *input, double *output, int N){
#pragma omp target teams distribute parallel for
	for (int i = 1; i < N-1; i++) {
	    output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0;
	}
}

void atomic_add(double *counter, int N) {
#pragma omp target teams distribute parallel for
	for (int i = 0; i < N; i++) {
	    #pragma omp atomic
	    counter++;
	}
}

void coalesced_access(short *data, int N){
#pragma omp target teams distribute parallel for
	for (int i = 0; i < N; i++) {
	    data[i] += 1.0;  // Coalesced access
	}
}

void uncoal_access(short *data, int N, int stride){
#pragma omp target teams distribute parallel for
	for (int i = 0; i < N; i += stride) {
	    data[i] += 1.0;  // Uncoalesced access
	}
}

void register_spill(double *data, int N) {
#pragma omp target teams distribute parallel for
	for (int ii = 0; ii < N; ii++) {
	    double a, b, c, d, e, f, g, h, i, j;
	    a = data[ii];
	    b = a * 2;
	    c = b * 2;
	    d = c * 2;
	    e = d * 2;
	    f = e * 2;
	    g = f * 2;
	    h = g * 2;
	    i = h * 2;
	    j = i * 2;
	    data[ii] = j;  
	}
}


void branch_divergence(double *data, int N) {
#pragma omp target teams distribute parallel for
	for (int i = 0; i < N; i++) {
	    if (i % 2 == 0) {  // Divergent branch
		data[i] += 1;
	    } else {
		data[i] -= 1;
	    }
	}
}

void uniform_branch(double *data, int N) {
#pragma omp target teams distribute parallel for
	for (int i = 0; i < N; i++) {
	    if (i < N) {  // Uniform branch
		data[i] += 1;
	    } 
	}
}

void multiple_access_not_cached(double *data, int N) {
#pragma omp target teams distribute parallel for num_threads(256) num_teams(114688)
        for (int i = 0; i < N; i++) {
            if (i < N) {  // Uniform branch
                data[i] += data[N-i -1];
            }
        }
}

int main(){
  // alloc on host
  double* a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  double* c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*SIZE);
  for (int i = 0; i < SIZE; i++)
  {
    a[i] = i;
    b[i] = i+1;
  }
  // alloc on device
#pragma omp target enter data map(to: a[0:SIZE], b[0:SIZE])
#pragma omp target enter data map(alloc:c[0:SIZE])


  vec_add(a,b,c);
  stencil_1d(a,c,SIZE);
  atomic_add(c,SIZE);
  coalesced_access((short*)c, SIZE*4);
  uncoal_access((short*)c, SIZE*4, 16);
  register_spill(c, SIZE);
  branch_divergence(c, SIZE);
  uniform_branch(c, SIZE);
  multiple_access_not_cached(c, SIZE);
#pragma omp target update from(c[0:SIZE])
#pragma omp target exit data map(release: a[0:SIZE], b[0:SIZE], c[0:SIZE])
  printf("c0:%f ", c[0] );


}
