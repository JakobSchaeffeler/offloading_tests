#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

#define SIZE 2048 //8192
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024
#define INDEX(x, y, N) ((x) + (y) * (N))

__global__ void stencil2d(double *in, double *out, int N) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (x > 0 && x < N-1 && y > 0 && y < N-1) {
        out[INDEX(x, y, N)] = (in[INDEX(x-1, y, N)] +
                               in[INDEX(x+1, y, N)] +
                               in[INDEX(x, y-1, N)] +
                               in[INDEX(x, y+1, N)]) / 4.0f;
    }
}

__global__ void mat_mul(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}



int main(){
	double* hA = (double*) malloc(SIZE*SIZE*sizeof(double));
    	double* hB = (double*) malloc(SIZE*SIZE*sizeof(double));
	double* hC = (double*) malloc(SIZE*SIZE*sizeof(double));
	    
	double *dA, *dB, *dC;
   	hipMalloc(&dA, SIZE*SIZE * sizeof(double));
   	hipMalloc(&dB, SIZE*SIZE * sizeof(double));
    	hipMalloc(&dC, SIZE*SIZE * sizeof(double));	
	
    for (int i = 0; i < SIZE; i++){
	    for (int j = 0; j < SIZE; j++){
            hA[i*SIZE + j] = i*SIZE + j;
            hB[i*SIZE + j] = i*SIZE -j;
            hC[i*SIZE + j] = 0;
  	    }
    }
	hipMemcpy(dA, hA, SIZE*SIZE * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, SIZE*SIZE * sizeof(double), hipMemcpyHostToDevice);
  	hipMemcpy(dC, hC, SIZE*SIZE * sizeof(double), hipMemcpyHostToDevice);

	for(int i = 0; i < 1; i++){
		dim3 threadsPerBlock(32, 32);
		dim3 numBlocks((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,(SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

		hipLaunchKernelGGL(stencil2d, numBlocks, threadsPerBlock, 0, 0, dA, dB, SIZE);
		
		hipLaunchKernelGGL(mat_mul, numBlocks, threadsPerBlock, 0, 0, dA, dB, dC, SIZE);
		
	}

	hipMemcpy(hC, dC, SIZE * SIZE * sizeof(double), hipMemcpyDeviceToHost);

	// Print the first result
	printf("c0: %f\n", hC[0]);
	printf("c_last %f\n", hC[SIZE*SIZE-1]);
	// Free memory
	free(hA);
	free(hB);
	free(hC);
	hipFree(dB);
	hipFree(dB);
	hipFree(dC);

	return 0;



}
