#include <cstdlib>
#include <iostream>

#define SIZE 33554432
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024
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

int main(){
	double* hc = (double*) malloc(SIZE*sizeof(double));
	    
	double *dc;
	cudaMalloc(&dc, SIZE*sizeof(double));
	
	for(int i = 0; i < SIZE; i++){
		hc[i] = i;
	}

	cudaMemcpy(dc, hc, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	
	register_spill<<<SIZE/TBSIZE, TBSIZE>>>(dc, SIZE);

	cudaMemcpy(hc, dc, SIZE*sizeof(double), cudaMemcpyDeviceToHost);

	printf("c0: %f", hc[0] );


}
