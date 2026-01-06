#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
const int bM = 16;
const int bN = 16;
const int bK = 16;
const int N = 512;
const int K = 512;
const int M = 512;
void initialize_matrix(float *matrix, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      matrix[i * N + j] = rand() % 100;
    }
  }
}

void print_matrix(const float *matrix, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf("%.0f ", matrix[i * N + j]);
    }
    printf("\n");
  }
}

void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0;
      for (int k = 0; k < K; k++)
      {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

bool check_result(const float *C_cpu, const float *C_cuda, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      if (C_cpu[i * N + j] - C_cuda[i * N + j] > 1e-6)
      {
        std::cout << "Error at position (" << i << ", " << j << ")" << std::endl;
        std::cout << "CPU value: " << C_cpu[i * N + j] << ", CUDA value: " << C_cuda[i * N + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

__global__ void baseline_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float sum = 0;
  for (int k = 0; k < K; k++)
  {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}


__global__ void tiled_gemm(
  const float *A, 
  const float *B, 
  float *C, 
  int M, 
  int N, 
  int K)
{
  // 使用 extern 声明动态共享内存
  // tileA: [bM][K], tileB: [bN][K] - 不拆分K维度
  extern __shared__ float shared_mem[];
  float *tileA = shared_mem;
  float *tileB = shared_mem + bM * K;
  
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int tM = threadIdx.y;
  int tN = threadIdx.x;
  
  // 协作加载 tileA: 每个block加载 bM 行，每行 K 个元素
  // 每个线程加载自己对应的那一行，但需要加载多个元素来完成整个K维度
  int elements_per_thread_A = (K + blockDim.x - 1) / blockDim.x;
  for (int e = 0; e < elements_per_thread_A; e++) {
    int k_idx = e * blockDim.x + tN;
    if (k_idx < K) {
      if (row < M) {
        tileA[tM * K + k_idx] = A[row * K + k_idx];
      } else {
        tileA[tM * K + k_idx] = 0.0f;  // 边界情况填充0
      }
    }
  }
  
  // 协作加载 tileB: 每个block加载 bN 列，每列 K 个元素
  // tileB[tN] 存储 B 的第 col 列的所有 K 个元素
  // 每个线程(tM, tN)负责加载 tileB[tN] 的部分元素（使用tM来分布K维度）
  int elements_per_thread_B = (K + blockDim.y - 1) / blockDim.y;
  for (int e = 0; e < elements_per_thread_B; e++) {
    int k_idx = e * blockDim.y + tM;
    if (k_idx < K) {
      if (col < N) {
        tileB[tN * K + k_idx] = B[k_idx * N + col];
      } else {
        tileB[tN * K + k_idx] = 0.0f;  // 边界情况填充0
      }
    }
  }

  __syncthreads();
  // 计算：C[row][col] = sum(k) tileA[tM][k] * tileB[tN][k]
  // tileA[tM][k] 存储的是 A[row][k]
  // tileB[tN][k] 存储的是 B[k][col]
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += tileA[tM * K + k] * tileB[tN * K + k];
    }
    C[row * N + col] = sum;
  }
}


int main(int argc, char *argv[])
{
  int option = 0;
  if (argc > 1) {
    option = std::stoi(argv[1]);
  } else {
    option = 0;
  }
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  float *C_cpu = new float[M * N];
  initialize_matrix(A, M, K);
  initialize_matrix(B, K, N);
  
  // 测量 CPU GEMM 运行时间（多次运行取平均值）
  int warmup_runs = 3;
  int measure_runs = 10;
  double cpu_time_total = 0.0;

  
  // 精确测量 CPU 时间
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++) {
    cpu_gemm(A, B, C_cpu, M, N, K);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
  cpu_time_total = cpu_duration.count() / 10.0;
  
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, M * K * sizeof(float));
  cudaMalloc((void**)&d_B, K * N * sizeof(float));
  cudaMalloc((void**)&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
  // 创建 CUDA Events 用于精确测量 GPU 时间
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // 注意：dim3 的顺序是 (x, y)，在 GEMM 中通常 x 对应列(N)，y 对应行(M)
  dim3 blocksize(bN, bM);  // blockDim.x = bN (列), blockDim.y = bM (行)
  dim3 gridsize((N + blocksize.x - 1) / blocksize.x, (M + blocksize.y - 1) / blocksize.y);
  
  // 检查 GPU 共享内存限制
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int shared_size_required = (bM * K + bN * K) * sizeof(float);
  
  printf("GPU共享内存限制:\n");
  printf("  sharedMemPerBlock: %zu 字节 (%.2f KB)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0f);
  printf("  sharedMemPerBlockOptin: %zu 字节 (%.2f KB)\n", prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0f);
  printf("所需共享内存: %d 字节 (%.2f KB)\n", shared_size_required, shared_size_required / 1024.0f);
  
  // 如果使用 tiled_gemm，需要设置更大的共享内存限制
  if (option == 1) {
    if (shared_size_required > prop.sharedMemPerBlockOptin) {
      printf("错误: 所需共享内存 (%d 字节) 超过GPU可选限制 (%zu 字节)!\n", 
             shared_size_required, prop.sharedMemPerBlockOptin);
      printf("无法使用 tiled_gemm，请使用 baseline_gemm (option=0)\n");
      return 1;
    }
    
    // 设置动态共享内存属性，允许使用 sharedMemPerBlockOptin
    if (shared_size_required > prop.sharedMemPerBlock && shared_size_required <= prop.sharedMemPerBlockOptin) {
      // 设置最大动态共享内存大小，允许使用 Opt-in 共享内存
      cudaFuncSetAttribute(tiled_gemm, 
                          cudaFuncAttributeMaxDynamicSharedMemorySize, 
                          prop.sharedMemPerBlockOptin);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("警告: 设置共享内存属性失败: %s\n", cudaGetErrorString(err));
        printf("将尝试直接启动，可能失败\n");
      } else {
        printf("已设置最大动态共享内存为: %zu 字节 (%.2f KB)\n", 
               prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0f);
      }
    }
  }
  
  // 预热 GPU
  for (int i = 0; i < warmup_runs; i++) {
    if (option == 0) {
      baseline_gemm<<<gridsize, blocksize>>>(d_A, d_B, d_C, M, N, K);
    } else if (option == 1) {
      // 计算共享内存大小: tileA[bM][K] + tileB[bN][K] - 不拆分K维度
      int shared_size = (bM * K + bN * K) * sizeof(float);
      if (shared_size > prop.sharedMemPerBlockOptin) {
        printf("错误: 共享内存大小超过可选限制，无法启动核函数\n");
        return 1;
      }
      tiled_gemm<<<gridsize, blocksize, shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
  }
  cudaDeviceSynchronize(); // 确保预热完成
  
  // 精确测量 GPU 核函数时间
  float gpu_time_total = 0.0f;
  cudaEventRecord(start);
  for (int i = 0; i < measure_runs; i++) {
    if (option == 0) {
      baseline_gemm<<<gridsize, blocksize>>>(d_A, d_B, d_C, M, N, K);
    } else if (option == 1) {
      // 计算共享内存大小: tileA[bM][K] + tileB[bN][K] - 不拆分K维度
      int shared_size = (bM * K + bN * K) * sizeof(float);
      // 已经在之前检查过，这里直接使用
      tiled_gemm<<<gridsize, blocksize, shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop); // 等待所有 CUDA 操作完成
  
  float gpu_time_ms = 0.0f;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  gpu_time_total = gpu_time_ms * 1000.0f / (float)measure_runs; // 转换为微秒
  
  // 检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
  }
  
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  
  // 检查内存拷贝错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error after memcpy: %s\n", cudaGetErrorString(err));
  }
  
  // 输出时间测量结果
  printf("\n========== 性能测量结果 ==========\n");
  printf("矩阵尺寸: M=%d, N=%d, K=%d\n", M, N, K);
  printf("测量次数: %d 次运行的平均值\n", measure_runs);
  printf("CPU GEMM 运行时间: %.3f 微秒 (%.3f 毫秒)\n", cpu_time_total, cpu_time_total / 1000.0);
  printf("GPU GEMM 运行时间: %.3f 微秒 (%.3f 毫秒)\n", gpu_time_total, gpu_time_total / 1000.0);
  printf("加速比: %.2fx\n", cpu_time_total / gpu_time_total);
  printf("===================================\n\n");
  
  if (!check_result(C_cpu, C, M, N)) {
    printf("Error: result check failed\n");
  } else {
    printf("Result check passed\n");
  }
  
  // 清理 CUDA Events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_cpu;
  return 0;
}