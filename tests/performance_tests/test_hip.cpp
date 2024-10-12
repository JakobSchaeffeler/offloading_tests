#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024


__global__ void vec_add(double *a, double *b, double *c) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < SIZE) {
        c[idx] = a[idx] + b[idx];
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

__global__ void stencil_1d(double *input, double *output, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i > 0 && i < N - 1) {
        output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0;
    }
}
/*
__global__ void atomic_add(double *counter, int N) {
        atomicAdd(counter, (double)1.0);
}
*/

__global__ void coalesced_access(short *data, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        data[idx] += 1.0;  
    }
}

__global__ void uncoal_access(short *data, int N, int stride) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx * stride < N) {
        data[idx * stride] += 1.0;  
    }
}

__global__ void register_spill(double *data, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) {
        // Local array to force register spilling
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

        // Store the result back to global memory
        data[i] = sum;
   }
}
   
__global__ void branch_divergence(double *data, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
	    if (idx % 2 == 0) {  // Divergent branch
		data[idx] += 1;
	    } else {
		data[idx] -= 1;
	    }
    }
}

__global__ void uniform_branch(double *data, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
	    if (idx < N * 2) {  // No divergence (all threads in warp follow same path)
		data[idx] += 1;
	    }
    }
}

__global__ void multiple_access_not_cached(double *data, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
	    data[idx] += data[N-1-idx];
    }
}


int main(){
	double* ha = (double*) malloc(SIZE*sizeof(double));
        double* hb = (double*) malloc(SIZE*sizeof(double));
	double* hc = (double*) malloc(SIZE*sizeof(double));
	    
	double *da, *db, *dc;
   	hipMalloc(&da, SIZE * sizeof(double));
   	hipMalloc(&db, SIZE * sizeof(double));
    	hipMalloc(&dc, SIZE * sizeof(double));	
	
	for(int i = 0; i < SIZE; i++){
		ha[i] = i;
		hb[i] = i+1;
	}

	hipMemcpy(da, ha, SIZE * sizeof(double), hipMemcpyHostToDevice);
    	hipMemcpy(db, hb, SIZE * sizeof(double), hipMemcpyHostToDevice);
  	
	for(int i = 0; i < 10; i++){
		
		hipLaunchKernelGGL(vec_add, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, da, db, dc);
		
		hipLaunchKernelGGL(stencil_1d, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, da, dc, SIZE);
		
		// hipLaunchKernelGGL(atomic_add, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);

		hipLaunchKernelGGL(coalesced_access, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, (short*)dc, SIZE * 4);
		
		hipLaunchKernelGGL(uncoal_access, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, (short*)dc, SIZE * 4, 16);

		//hipLaunchKernelGGL(register_spill, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);

		hipLaunchKernelGGL(branch_divergence, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);

		hipLaunchKernelGGL(uniform_branch, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);

		hipLaunchKernelGGL(multiple_access_not_cached, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);
	}
	// Copy results back to host
	hipMemcpy(hc, dc, SIZE * sizeof(double), hipMemcpyDeviceToHost);

	// Print the first result
	printf("c0: %f", hc[0]);

	// Free memory
	free(ha);
	free(hb);
	free(hc);
	hipFree(da);
	hipFree(db);
	hipFree(dc);

	return 0;



}
