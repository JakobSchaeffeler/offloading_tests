#include <omp.h>
#include <stdio.h>
#define SIZE 29360128


int main() {

    double* array; //= (double*) aligned_alloc(64, sizeof(double)*N);
    double* result; //= (double*) aligned_alloc(64, sizeof(double)*N);
    // Allocate memory on the device for array and result
    #pragma omp target enter data map(alloc: array[0:SIZE], result[0:SIZE])

    // Initialize the array directly on the device
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE; i++) {
        array[i] = i;  // Initialize array[i] on the device
    }
    // Offload the computation to the GPU
    #pragma omp target teams distribute parallel for 
    for (int i = 0; i < SIZE; i++) {
        // Each thread accesses a consecutive element of the array
        result[i] = array[i] * 2;
    }
    
    #pragma omp target teams distribute parallel for 
    for (int i = 0; i < SIZE; i++) {
        if (result[i] != array[i] * 2) {
            printf("Error at index %d\n", i);
        }
    }
    #pragma omp target exit data map(release: array[0:SIZE], result[0:SIZE])
}
