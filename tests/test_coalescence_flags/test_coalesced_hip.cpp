#include <cstdlib> 
#include <iostream>
#include "hip/hip_runtime.h"

#define SIZE 29360128
#define ALIGNMENT (2*1024*1024)
#define TBSIZE 1024

std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  hipSetDevice(device);
  check_error();
  int driver;
  hipDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

__global__ void triad_kernel(double* a, const double* b, const double* c)
{
  const double scalar = 1.5;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = b[i] + scalar * c[i];
}

void triad(double* a, const double* b, const double* c)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel), dim3(SIZE/TBSIZE), dim3(TBSIZE), 0, 0, a, b, c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}


__global__ void init_kernel(double* b, double* c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  b[i] = i;
  c[i] = 2*i;
}

void init_arrays(double* b, double* c)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(init_kernel), dim3(SIZE/TBSIZE), dim3(TBSIZE), 0, 0, b, c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}


int main(){
  // The array size must be divisible by TBSIZE for kernel launches
  if (SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  int device_index = 0;
  hipGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  hipSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using HIP device " << getDeviceName(0) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(0) << std::endl;

  int array_size = SIZE;

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*SIZE*sizeof(double))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  double* d_a;
  double* d_b;
  double* d_c;

  hipMalloc(&d_a, SIZE*sizeof(double));
  check_error();
  hipMalloc(&d_b, SIZE*sizeof(double));
  check_error();
  hipMalloc(&d_c, SIZE*sizeof(double));
  check_error();

  init_arrays(d_b, d_c);

  triad(d_a, d_b, d_c);

  double* a = (double*)malloc(SIZE*sizeof(double));

  double* h_a = (double*)malloc(SIZE*sizeof(double));

  hipMemcpy(a, d_a, SIZE*sizeof(double), hipMemcpyDeviceToHost);

  double sum = 0;
  double sum_wanted = 0;
  for (int i = 0; i < SIZE; i++){
    sum += a[i];
    sum_wanted += i + 2*i*1.5;
  }
  if (sum != sum_wanted){
    std::cout << "Error in coalesced hip test" << std::endl;
  }
  

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();

}


