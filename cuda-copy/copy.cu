#include<iostream>
#include <cstdlib>
#include <string>
#include "TimeElapsed.h"

// Print usage information
void printUsage(const char* programName) {
  std::cout << "Usage: " << programName << " <n>" << std::endl;
  std::cout << "  n: Number of elements to copy (must be a positive integer)" << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << programName << " 1000000" << std::endl;
}

// Check CUDA error and exit if failed
void checkCudaError(cudaError_t err, const char* operation) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

//baseline copy kernel
__global__ void copy_baseline(float* src, float* dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

int main(int argc, char** argv) {
  // Check argument count
  if (argc != 2) {
    std::cerr << "Error: Invalid number of arguments." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Parse and validate input parameter
  int n = 0;
  try {
    n = std::stoi(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << "Error: Invalid input parameter. '" << argv[1] << "' is not a valid integer." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Validate that n is positive
  if (n <= 0) {
    std::cerr << "Error: Number of elements must be a positive integer (got " << n << ")." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }
  //create host arrays
  float* host_src = new float[n];
  float* host_dst = new float[n];

  //initialize host arrays
  for (int i = 0; i < n; i++) {
    host_src[i] = i;
  }

  //copy host src to host dst
  for (int i = 0; i < n; i++) {
    host_dst[i] = host_src[i];
  }

  float* device_src = nullptr;
  float* device_dst = nullptr;
  checkCudaError(cudaMalloc((void**)&device_src, n * sizeof(float)), "cudaMalloc (device_src)");
  checkCudaError(cudaMalloc((void**)&device_dst, n * sizeof(float)), "cudaMalloc (device_dst)");
  checkCudaError(cudaMemcpy(device_src, host_src, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy (HostToDevice)");

  dim3 block_size(1024);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);
  
  {
    RECORD_START();
    // Launch kernel
    copy_baseline<<<grid_size, block_size>>>(device_src, device_dst, n);
    RECORD_STOP();
  }
  
  // Check for kernel launch errors
  checkCudaError(cudaGetLastError(), "kernel launch");
  // Synchronize to ensure kernel completes
  checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  
  checkCudaError(cudaMemcpy(host_dst, device_dst, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy (DeviceToHost)");

  for (int i = 0; i < n; i++) {
    if (host_dst[i] != host_src[i]) {
      std::cout << "Error: host_dst[" << i << "] = " << host_dst[i] << " != host_src[" << i << "] = " << host_src[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Copy successful" << std::endl;
  //free device arrays
  cudaFree(device_src);
  cudaFree(device_dst);
  delete[] host_src;
  delete[] host_dst;
  return 0;
}