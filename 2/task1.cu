#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define IDX2C(row, col, cols) ((row) * (cols) + (col))

// Matrix dimensions: A is M x K, B is K x N, C is M x N
const int M = 10000;
const int K = 10000;
const int N = 10000;

// Simple CUDA error check
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Fill array with random floats in [0,1)
void fillRandom(float *a, int len) {
  for (int i = 0; i < len; ++i) {
    a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

// CPU reference: C = A * B
void matmulCPU(const float *A, const float *B, float *C,
               int m, int k, int n) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += A[IDX2C(row, p, k)] * B[IDX2C(p, col, n)];
      }
      C[IDX2C(row, col, n)] = sum;
    }
  }
}

// Max absolute difference between two arrays
float maxAbsDiff(const float *a, const float *b, int len) {
  float maxDiff = 0.0f;
  for (int i = 0; i < len; ++i) {
    float diff = fabsf(a[i] - b[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  return maxDiff;
}

// Task 1 kernel: one thread computes one C[row, col] using ONLY global memory
__global__
void matmulKernel(const float *A, const float *B, float *C, int m, int k, int n) {
  // Map thread to output coordinates
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → columns (n)
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → rows (m)

  // Guard against threads that fall outside the matrix
  if (row >= m || col >= n) return;

  float sum = 0.0f;

  // Inner loop over k: dot product of row of A and column of B
  for (int p = 0; p < k; ++p) {
    float a_val = A[IDX2C(row, p, k)];  // A[row, p]
    float b_val = B[IDX2C(p, col, n)];  // B[p, col]
    sum += a_val * b_val;
  }

  // Write result
  C[IDX2C(row, col, n)] = sum;
}

int main() {
  srand(67);

  int m = M, k = K, n = N;

  size_t sizeA = static_cast<size_t>(m) * k * sizeof(float);
  size_t sizeB = static_cast<size_t>(k) * n * sizeof(float);
  size_t sizeC = static_cast<size_t>(m) * n * sizeof(float);

  // Host allocations
  float *h_A    = (float*)malloc(sizeA);
  float *h_B    = (float*)malloc(sizeB);
  float *h_C    = (float*)malloc(sizeC);      // GPU result
  float *h_Cref = (float*)malloc(sizeC);      // CPU reference

  if (!h_A || !h_B || !h_C || !h_Cref) {
    fprintf(stderr, "Host malloc failed\n");
    return EXIT_FAILURE;
  }

  // Initialize A, B
  fillRandom(h_A, m * k);
  fillRandom(h_B, k * n);

  // For correctness testing on small sizes; skip on huge matrices
  bool doVerify = (m <= 1024 && k <= 1024 && n <= 1024);
  if (doVerify) {
    matmulCPU(h_A, h_B, h_Cref, m, k, n);
  } else {
    printf("Skipping CPU verification for large size (%d x %d * %d x %d)\n",
           m, k, k, n);
  }

  // Device allocations
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  cudaError_t err;
  err = cudaMalloc((void**)&d_A, sizeA);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }
  err = cudaMalloc((void**)&d_B, sizeB);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    return EXIT_FAILURE;
  }
  err = cudaMalloc((void**)&d_C, sizeC);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    return EXIT_FAILURE;
  }

  // Copy inputs to device
  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  // Loop over block sizes for timing
  int blockSizes[3] = {8, 16, 32};

  for (int i = 0; i < 3; ++i) {
    int BS = blockSizes[i];

    dim3 block(BS, BS);
    dim3 grid(
      (n + block.x - 1) / block.x,
      (m + block.y - 1) / block.y
    );

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timed launch
    cudaEventRecord(start);
    matmulKernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCUDAError("matmulKernel");

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // GFLOPs
    double ops    = 2.0 * (double)m * (double)k * (double)n;
    double secs   = elapsedMs / 1000.0;
    double gflops = ops / (secs * 1.0e9);

    printf("Block %dx%d -> Time: %.3f ms, GFLOPs: %.2f\n", BS, BS, elapsedMs, gflops);
  }

  // Optional: correctness check when size is small (single block size)
  if (doVerify) {
    // For verification, just reuse the last BS (32) or set a fixed one:
    dim3 block(16, 16);
    dim3 grid(
      (n + block.x - 1) / block.x,
      (m + block.y - 1) / block.y
    );

    matmulKernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    checkCUDAError("verification matmulKernel");

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    float maxDiff = maxAbsDiff(h_C, h_Cref, m * n);
    printf("Max abs diff between CPU and GPU: %e\n", maxDiff);
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Cref);

  return 0;
}