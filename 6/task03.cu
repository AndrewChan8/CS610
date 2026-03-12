#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>
#include <iostream>

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void generate_data(float * data, unsigned int length) {
  for (unsigned int i = 0; i < length; ++i) {
    data[i] = (float)rand() / RAND_MAX;
  }
  std::sort(data, data + length);
}

__device__ int co_rank(int k, const float *A, int m, const float *B, int n) {
  int i_low = max(0, k - n);
  int i_high = min(k, m);

  while (i_low < i_high) {
    int i = (i_low + i_high) / 2;
    int j = k - i;

    if (i < m && j > 0 && A[i] < B[j - 1]) {
      i_low = i + 1;
    } else if (i > 0 && j < n && A[i - 1] > B[j]) {
      i_high = i - 1;
    } else {
      return i;
    }
  }

  int i = i_low;
  int j = k - i;

  if (i > 0 && j < n && A[i - 1] > B[j]) {
    i--;
  }

  return i;
}

__global__ void GPU_merge(float * A, unsigned int m, float * B, unsigned int n, float * C, unsigned int tile_size) {
  extern __shared__ float shared[];
  float *sA = shared;
  float *sB = shared + tile_size;

  unsigned int block_tile_start = blockIdx.x * tile_size;
  unsigned int block_tile_end = min(block_tile_start + tile_size, m + n);
  unsigned int c_len = block_tile_end - block_tile_start;

  if (c_len == 0) return;

  int a_start = co_rank(block_tile_start, A, m, B, n);
  int b_start = block_tile_start - a_start;

  int a_end = co_rank(block_tile_end, A, m, B, n);
  int b_end = block_tile_end - a_end;

  int a_len = a_end - a_start;
  int b_len = b_end - b_start;

  int tid = threadIdx.x;

  if (tid < a_len) {
    sA[tid] = A[a_start + tid];
  }
  if (tid < b_len) {
    sB[tid] = B[b_start + tid];
  }
  __syncthreads();

  if (tid < c_len) {
    int local_k = tid;
    int local_i = co_rank(local_k, sA, a_len, sB, b_len);
    int local_j = local_k - local_i;

    float a_val = (local_i < a_len) ? sA[local_i] : INFINITY;
    float b_val = (local_j < b_len) ? sB[local_j] : INFINITY;

    C[block_tile_start + tid] = (a_val <= b_val) ? a_val : b_val;
  }
}

void CPU_merge(float * A, unsigned int m, float * B, unsigned int n, float * C) {
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }

  while (i < m) {
    C[k++] = A[i++];
  }

  while (j < n) {
    C[k++] = B[j++];
  }
}

/* Host code */
int main(void) {
  unsigned int input_length_A = 20;
  unsigned int input_length_B = 20;
  unsigned int output_length_C = input_length_A + input_length_B;

  float * h_A, * d_A, * h_B, * d_B, * h_C, * d_C;
  cudaEvent_t start, stop;
  float ms;

  unsigned int data_size_A = input_length_A * sizeof(float);
  unsigned int data_size_B = input_length_B * sizeof(float);
  unsigned int data_size_C = output_length_C * sizeof(float);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc((void**)&d_A, data_size_A);
  cudaMalloc((void**)&d_B, data_size_B);
  cudaMalloc((void**)&d_C, data_size_C);
  checkCUDAError("CUDA malloc");

  h_A = (float*)malloc(data_size_A);
  h_B = (float*)malloc(data_size_B);
  h_C = (float*)malloc(data_size_C);

  generate_data(h_A, input_length_A);
  generate_data(h_B, input_length_B);

  cudaMemcpy(d_A, h_A, data_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, data_size_B, cudaMemcpyHostToDevice);
  checkCUDAError("CUDA memcpy to device");

  unsigned int tile_size = 32;
  int blockSize = tile_size;
  int gridSize = (output_length_C + tile_size - 1) / tile_size;
  size_t sharedMemSize = 2 * tile_size * sizeof(float);

  cudaEventRecord(start, 0);

  GPU_merge<<<gridSize, blockSize, sharedMemSize>>>(d_A, input_length_A, d_B, input_length_B, d_C, tile_size);
  checkCUDAError("GPU_merge kernel");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  checkCUDAError("kernel timing");

  cudaMemcpy(h_C, d_C, data_size_C, cudaMemcpyDeviceToHost);
  checkCUDAError("CUDA memcpy from device");

  printf("Execution time:\t%f ms\n", ms);

  float * cpu_C = (float*)malloc(data_size_C);
  CPU_merge(h_A, input_length_A, h_B, input_length_B, cpu_C);

  int correct = 1;
  for (unsigned int i = 0; i < output_length_C; ++i) {
    if (fabs(h_C[i] - cpu_C[i]) > 1e-6f) {
      printf("Mismatch at index %u: GPU=%f CPU=%f\n", i, h_C[i], cpu_C[i]);
      correct = 0;
      break;
    }
  }

  if (correct) {
    printf("Merge verification: PASSED\n");
  } else {
    printf("Merge verification: FAILED\n");
  }

  free(cpu_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}