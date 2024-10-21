#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

#define SIZE 33554432
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024

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

int main(){
	double* hc = (double*) malloc(SIZE*sizeof(double));
	    
	double *dc;
    	hipMalloc(&dc, SIZE * sizeof(double));	
	
	for(int i = 0; i < SIZE; i++){
		hc[i] = i;
	}

  	hipMemcpy(dc, hc, SIZE * sizeof(double), hipMemcpyHostToDevice);
	
	hipLaunchKernelGGL(register_spill, dim3(SIZE / TBSIZE), dim3(TBSIZE), 0, 0, dc, SIZE);
	// Copy results back to host
	hipMemcpy(hc, dc, SIZE * sizeof(double), hipMemcpyDeviceToHost);

	// Print the first result
	printf("c0: %f", hc[0]);

	// Free memory
	free(hc);
	hipFree(dc);

	return 0;



}
