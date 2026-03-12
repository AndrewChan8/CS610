#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

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
}

__global__ void GPU_scan(float * X, float * Y, unsigned int length) {
  __shared__ float temp[2048];

  int tid = threadIdx.x;
  int i1 = tid;
  int i2 = tid + blockDim.x;

  if (i1 < length) temp[i1] = X[i1];
  if (i2 < length) temp[i2] = X[i2];
  __syncthreads();

  for (unsigned int stride = 1; stride < length; stride <<= 1) {
    float val1 = 0.0f, val2 = 0.0f;

    if (i1 < length && i1 >= stride)
      val1 = temp[i1 - stride];

    if (i2 < length && i2 >= stride)
      val2 = temp[i2 - stride];

    __syncthreads();

    if (i1 < length && i1 >= stride)
      temp[i1] += val1;

    if (i2 < length && i2 >= stride)
      temp[i2] += val2;

    __syncthreads();
  }

  if (i1 < length) Y[i1] = temp[i1];
  if (i2 < length) Y[i2] = temp[i2];
}

void CPU_scan(float * X, float * Y, unsigned int length) {
  if (length == 0) return;

  Y[0] = X[0];
  for (unsigned int i = 1; i < length; ++i) {
    Y[i] = Y[i - 1] + X[i];
  }
}

/* Host code */
int main(void) {
  unsigned int input_length = 2048;
  float * h_input, * d_input, * h_output, * d_output;
  cudaEvent_t start, stop;
  float ms;

  unsigned int data_size = input_length * sizeof(float);

  // create timers
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on the GPU
  cudaMalloc((void**)&d_input, data_size);
  cudaMalloc((void**)&d_output, data_size);
  checkCUDAError("CUDA malloc");

  // allocate host data
  h_input = (float*)malloc(data_size);
  h_output = (float*)malloc(data_size);
  generate_data(h_input, input_length);

  // copy input to device memory
  cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
  checkCUDAError("CUDA memcpy to device");
  
  int blockSize = 1024;
  int gridSize = 1;

  cudaEventRecord(start, 0);

  GPU_scan<<<gridSize, blockSize>>>(d_input, d_output, input_length);
  checkCUDAError("GPU_scan kernel");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  checkCUDAError("kernel timing");

  // copy output back from the GPU
  cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
  checkCUDAError("CUDA memcpy from device");

  // output timings
  printf("Execution time:\t%f ms\n", ms);

  // Verify output using a CPU function
  float * cpu_output = (float*)malloc(data_size);
  CPU_scan(h_input, cpu_output, input_length);

  int correct = 1;
  for (unsigned int i = 0; i < input_length; ++i) {
    if (fabs(h_output[i] - cpu_output[i]) > 1e-3f) {
      printf("Mismatch at index %u: GPU=%f CPU=%f\n",
             i, h_output[i], cpu_output[i]);
      correct = 0;
      break;
    }
  }

  if (correct) {
    printf("Scan verification: PASSED\n");
  } else {
    printf("Scan verification: FAILED\n");
  }

  // cleanup
  free(cpu_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  return 0;
}