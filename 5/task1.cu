#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 4194304
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void vectorAdd(int max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < max) {
    d_c[i] = d_a[i] + d_b[i];
  }
}

int main(void) {
  int *a, *b, *c, *c_ref;
  int errors = 0;
  unsigned int size = N * sizeof(int);

  // host allocations
  a     = (int *)malloc(size); random_ints(a);
  b     = (int *)malloc(size); random_ints(b);
  c     = (int *)malloc(size);
  c_ref = (int *)malloc(size);

  // copy inputs to static device globals
  cudaMemcpyToSymbol(d_a, a, size);
  cudaMemcpyToSymbol(d_b, b, size);
  checkCUDAError("cudaMemcpyToSymbol");

  // host reference
  for (unsigned int i = 0; i < N; ++i) {
    c_ref[i] = a[i] + b[i];
  }

  // timing setup
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  checkCUDAError("cudaEventCreate");

  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaEventRecord(start);
  vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(N);
  checkCUDAError("vectorAdd kernel launch");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCUDAError("cudaEventRecord/Sync");

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  checkCUDAError("cudaEventElapsedTime");

  // copy result back
  cudaMemcpyFromSymbol(c, d_c, size);
  checkCUDAError("cudaMemcpyFromSymbol");

  // correctness check
  for (unsigned int i = 0; i < N; ++i) {
    if (c[i] != c_ref[i]) {
      errors++;
    }
  }

  printf("VectorAdd errors: %d\n", errors);
  printf("Kernel time: %.3f ms\n", elapsed_ms);

  // measured bandwidth
  double total_bytes = 3.0 * (double)size;
  double seconds = elapsed_ms / 1000.0;
  double measured_bandwidth_GBps = (total_bytes / seconds) / 1.0e9;

  printf("Measured bandwidth: %.3f GB/s\n", measured_bandwidth_GBps);

  // theoretical bandwidth using device attributes
  int memClockKHz = 0;
  int memBusWidthBits = 0;

  cudaError_t attrErr1 = cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
  cudaError_t attrErr2 = cudaDeviceGetAttribute(&memBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);

  if (attrErr1 != cudaSuccess || attrErr2 != cudaSuccess){
    fprintf(stderr,
      "Error getting device memory attributes: %s / %s\n",
      cudaGetErrorString(attrErr1),
      cudaGetErrorString(attrErr2));
    exit(EXIT_FAILURE);
  }

  double memClockHz = (double)memClockKHz * 1000.0;
  double busWidthBytes = (double)memBusWidthBits / 8.0;

  double theoretical_GBps = 2.0 * memClockHz * busWidthBytes / 1.0e9;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  checkCUDAError("cudaGetDeviceProperties");

  printf("Device: %s\n", prop.name);
  printf("Theoretical memory bandwidth (approx): %.3f GB/s\n", theoretical_GBps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCUDAError("cudaEventDestroy");

  free(a);
  free(b);
  free(c);
  free(c_ref);

  return 0;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void random_ints(int *a) {
  for (unsigned int i = 0; i < N; i++) {
    a[i] = rand();
  }
}