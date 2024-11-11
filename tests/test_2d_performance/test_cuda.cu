#include <cstdlib>
#include <iostream>

#define SIZE 2048
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024
#define TILE_WIDTH 1024

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

/*
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


__global__ void mat_mul(double* A, double* B, double* C, int width) {
    // Allocate shared memory for sub-matrices
    __shared__ double sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column indexes
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    double sum = 0.0f;

    // Loop over sub-matrices
    for (int i = 0; i < width / TILE_WIDTH; i++) {
        // Load tiles into shared memory
        sA[threadIdx.y][threadIdx.x] = A[row * width + (i * TILE_WIDTH + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * width + col];

        __syncthreads();  // Ensure all threads have loaded their sub-matrix

        // Perform multiplication and accumulation
        for (int j = 0; j < TILE_WIDTH; j++) {
            sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        __syncthreads();  // Ensure all threads have completed the current iteration
    }

    // Write the result to the output matrix
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}
*/
__global__ void mat_mul(double *a, double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0;
    if (i < SIZE && j < SIZE) {
        for (int k = 0; k < n; k++)
            sum += a[i * SIZE + k] * b[k * SIZE + j];
        c[i * SIZE + j] = sum;
    }
}

int main(){
	double* hA = (double*) malloc(SIZE*SIZE*sizeof(double));
    	double* hB = (double*) malloc(SIZE*SIZE*sizeof(double));
	double* hC = (double*) malloc(SIZE*SIZE*sizeof(double));
	    
	double *dA, *dB, *dC;
	cudaMalloc(&dA, SIZE*SIZE*sizeof(double));
	cudaMalloc(&dB, SIZE*SIZE*sizeof(double));
	cudaMalloc(&dC, SIZE*SIZE*sizeof(double));
	
    for (int i = 0; i < SIZE; i++){
	    for (int j = 0; j < SIZE; j++){
		    hA[i*SIZE + j] = i*SIZE + j;
		    hB[i*SIZE + j] = i*SIZE -j;
		    hC[i*SIZE + j] = 0;
	    }
    }

	cudaMemcpy(dA, hA, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, hC, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);

	//dim3 threadsPerBlock(32,32);
	//dim3 numBlocks((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,(SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
	dim3 threadsPerBlock(32, 32);
   	//dim3 numBlocks((SIZE + 32 - 1) / 32, (SIZE + 32 - 1) / 32);
        int numBlocks = SIZE*SIZE/1024;
	//printf("thrd: %i, %i\n", threadsPerBlock.x,threadsPerBlock.y);
	//printf("team: %i, %i\n", numBlocks.x,numBlocks.y);

	stencil2d<<<numBlocks, threadsPerBlock>>>(dA, dB,SIZE);
	mat_mul<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, SIZE);
	cudaMemcpy(hC, dC, SIZE*sizeof(double), cudaMemcpyDeviceToHost);

	printf("c0: %f\n", hC[SIZE+1] );
        printf("c_last %f\n", hC[SIZE*(SIZE-1) + SIZE-1]);

	free(hA);
	free(hB);
	free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
