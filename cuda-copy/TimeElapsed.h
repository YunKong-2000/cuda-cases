#pragma once

#define RECORD_START() \
  cudaEvent_t start, stop; \
  cudaEventCreate(&start); \
  cudaEventCreate(&stop); \
  cudaEventRecord(start); \

#define RECORD_STOP() \
  cudaEventRecord(stop); \
  cudaEventSynchronize(stop); \
  float milliseconds = 0; \
  cudaEventElapsedTime(&milliseconds, start, stop); \
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl; \
  cudaEventDestroy(start); \
  cudaEventDestroy(stop); 
