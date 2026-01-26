#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define IDX2C(row, col, cols) ((row) * (cols) + (col))

// Matrix dimensions
const int M = 10000;
const int K = 10000;
const int N = 10000;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void fillRandom(float *a, int len) {
  for (int i = 0; i < len; ++i) {
    a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void matmulCPU(const float *A, const float *B, float *C, int m, int k, int n) {
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

float maxAbsDiff(const float *a, const float *b, int len) {
  float maxDiff = 0.0f;
  for (int i = 0; i < len; ++i) {
    float diff = fabsf(a[i] - b[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  return maxDiff;
}

__global__
void matmulTiledKernel(const float *A, const float *B, float *C, int m, int k, int n) {
  int TILE = blockDim.x;  // blockDim.x == blockDim.y

  extern __shared__ float shmem[];
  float *As = shmem;
  float *Bs = shmem + TILE * TILE;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float sum = 0.0f;

  int numTiles = (k + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {

    int aCol = t * TILE + tx;
    int bRow = t * TILE + ty;

    // Load A tile
    if (row < m && aCol < k)
      As[ty * TILE + tx] = A[IDX2C(row, aCol, k)];
    else
      As[ty * TILE + tx] = 0.0f;

    // Load B tile
    if (bRow < k && col < n)
      Bs[ty * TILE + tx] = B[IDX2C(bRow, col, n)];
    else
      Bs[ty * TILE + tx] = 0.0f;

    __syncthreads();

    // Compute partial dot product
    for (int kk = 0; kk < TILE; ++kk) {
      sum += As[ty * TILE + kk] * Bs[kk * TILE + tx];
    }

    __syncthreads();
  }

  if (row < m && col < n)
    C[IDX2C(row, col, n)] = sum;
}

int main() {
  srand(67);

  int m = M, k = K, n = N;

  size_t sizeA = (size_t)m * k * sizeof(float);
  size_t sizeB = (size_t)k * n * sizeof(float);
  size_t sizeC = (size_t)m * n * sizeof(float);

  float *h_A    = (float*)malloc(sizeA);
  float *h_B    = (float*)malloc(sizeB);
  float *h_C    = (float*)malloc(sizeC);
  float *h_Cref = (float*)malloc(sizeC);

  if (!h_A || !h_B || !h_C || !h_Cref) {
    fprintf(stderr, "Host malloc failed\n");
    return EXIT_FAILURE;
  }

  fillRandom(h_A, m * k);
  fillRandom(h_B, k * n);

  bool doVerify = (m <= 1024 && k <= 1024 && n <= 1024);
  if (doVerify) {
    matmulCPU(h_A, h_B, h_Cref, m, k, n);
  } else {
    printf("Skipping CPU verification for large size (%d x %d * %d x %d)\n", m, k, k, n);
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  int blockSizes[3] = {8, 16, 32};

  for (int i = 0; i < 3; ++i) {
    int BS = blockSizes[i];

    dim3 block(BS, BS);
    dim3 grid(
      (n + BS - 1) / BS,
      (m + BS - 1) / BS
    );

    size_t sharedBytes = 2 * BS * BS * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulTiledKernel<<<grid, block, sharedBytes>>>(d_A, d_B, d_C, m, k, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCUDAError("matmulTiledKernel");

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double ops    = 2.0 * (double)m * (double)k * (double)n;
    double secs   = elapsedMs / 1000.0;
    double gflops = ops / (secs * 1.0e9);

    printf("[Task2] Block %dx%d -> Time: %.3f ms, GFLOPs: %.2f\n", BS, BS, elapsedMs, gflops);
  }

  if (doVerify) {
    dim3 block(16, 16);
    dim3 grid(
      (n + block.x - 1) / block.x,
      (m + block.y - 1) / block.y
    );
    size_t sharedBytes = 2 * block.x * block.y * sizeof(float);

    matmulTiledKernel<<<grid, block, sharedBytes>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    checkCUDAError("verification kernel");

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    float maxDiff = maxAbsDiff(h_C, h_Cref, m * n);
    printf("[Task2] Max abs diff: %e\n", maxDiff);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Cref);

  return 0;
}
