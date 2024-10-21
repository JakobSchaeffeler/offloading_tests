#include <cstdlib>
#include <iostream>

#define SIZE 8192
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024

#define INDEX(x, y, N) ((x) + (y) * (N))

__global__ void stencil2d(double *in, double *out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < N-1 && y > 0 && y < N-1) {
        out[INDEX(x, y, N)] = (in[INDEX(x-1, y, N)] +
                               in[INDEX(x+1, y, N)] +
                               in[INDEX(x, y-1, N)] +
                               in[INDEX(x, y+1, N)]) / 4.0f;
    }
}

__global__ void mat_mul(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
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
	cudaMalloc(&dA, SIZE*sizeof(double));
	cudaMalloc(&dB, SIZE*sizeof(double));
	cudaMalloc(&dC, SIZE*sizeof(double));
	
    for (int i = 0; i < SIZE; i++){
	    for (int j = 0; j < SIZE; j++){
            hA[i*SIZE + j] = i*SIZE + j;
            hB[i*SIZE + j] = i*SIZE -j;
            hC[i*SIZE + j] = 0;
  	    }
    }

	cudaMemcpy(dA, hA, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);
	for(int i = 0; i < 1; i++){
		dim3 threadsPerBlock(32,32);
		dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,(N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        	stencil2d<<<numBlocks, threadsPerBlock>>>(dA, dB,SIZE);
		dim3 threadsPerBlock(32,32);
    		dim3 numBlocks((N + TBSIZE - 1) / TBSIZE, (N + TBSIZE - 1) / TBSIZE);
		mat_mul<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, SIZE);
	}
	cudaMemcpy(hC, dC, SIZE*sizeof(double), cudaMemcpyDeviceToHost);

	printf("c0: %f\n", hC[0] );
        printf("c_last %f\n", hC[SIZE*SIZE-1]);

	free(hA);
	free(hB);
	free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
