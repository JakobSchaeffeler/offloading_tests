#include <cstdlib> 
#include <iostream>
#include "hip/hip_runtime.h"

#define SIZE 29360128
#define ALIGNMENT 64//(2*1024*1024)
#define TBSIZE 256
void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

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
__global__ void triad_kernel(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c)
{
  const float scalar = 1.5;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = b[i] + scalar * c[i];
}

void triad(float* a, const float* b, const float* c)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel), dim3(SIZE/TBSIZE), dim3(TBSIZE), 0, 0, a, b, c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

__global__ void init_kernel(float* b, float* c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  b[i] = i;
  c[i] = 2*i;
}

void init_arrays(float* b, float* c)
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
  if (props.totalGlobalMem < 3*SIZE*sizeof(float))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  float* d_a;
  float* d_b;
  float* d_c;

  hipMalloc(&d_a, SIZE*sizeof(float));
  check_error();
  hipMalloc(&d_b, SIZE*sizeof(float));
  check_error();
  hipMalloc(&d_c, SIZE*sizeof(float));
  check_error();

  init_arrays(d_b, d_c);

  triad(d_a, d_b, d_c);

  float* a = (float*)malloc(SIZE*sizeof(float));

  float* h_a = (float*)malloc(SIZE*sizeof(float));

  hipMemcpy(a, d_a, SIZE*sizeof(float), hipMemcpyDeviceToHost);

  float sum = 0;
  float sum_wanted = 0;
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


