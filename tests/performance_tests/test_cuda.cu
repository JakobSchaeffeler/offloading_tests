#include <cstdlib>
#include <iostream>

#define SIZE 33554432
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024


__global__ void vec_add(double *a, double *b, double *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        c[idx] = a[idx] + b[idx];
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

__global__ void stencil_1d(double *input, double *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0;  
    }
}

__global__ void uncoal_access(short *data, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < N) {
        data[idx * stride] += 1.0;  
    }
}

__global__ void register_spill(double *data, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
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

__global__ void branch_divergence(double *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
	    if (idx % 2 == 0) {  // Divergent branch
		data[idx] += 1;
	    } else {
		data[idx] -= 1;
	    }
    }
}

__global__ void uniform_branch(double *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
	    if (idx < N * 2) {  // No divergence (all threads in warp follow same path)
		data[idx] += 1;
	    }
    }
}

__global__ void multiple_access_not_cached(double *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
	    data[idx] += data[N-1-idx];
    }
}


int main(){
	double* ha = (double*) malloc(SIZE*sizeof(double));
        double* hb = (double*) malloc(SIZE*sizeof(double));
	double* hc = (double*) malloc(SIZE*sizeof(double));
	    
	double *da, *db, *dc;
	cudaMalloc(&da, SIZE*sizeof(double));
	cudaMalloc(&db, SIZE*sizeof(double));
	cudaMalloc(&dc, SIZE*sizeof(double));
	
	for(int i = 0; i < SIZE; i++){
		ha[i] = i;
		hb[i] = i+1;
	}

	cudaMemcpy(da, ha, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	for(int i = 0; i < 100; i++){
		
		vec_add<<<SIZE/TBSIZE, TBSIZE>>>(da, db, dc);

		stencil_1d<<<SIZE/TBSIZE, TBSIZE>>>(da, dc, SIZE);
		
		atomic_add<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);
		
		coalesced_access<<<SIZE*4/TBSIZE, TBSIZE>>>((short*)dc, SIZE*4);
		
		uncoal_access<<<SIZE/TBSIZE, TBSIZE>>>((short*)dc, SIZE*4, 16);

		register_spill<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);

		branch_divergence<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);

		uniform_branch<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);
		
		multiple_access_not_cached<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);
	}
	cudaMemcpy(hc, dc, SIZE*sizeof(double), cudaMemcpyDeviceToHost);

	printf("c0: %f", hc[0] );


}
