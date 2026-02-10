#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define IDX2C(row, col, cols) ((row) * (cols) + (col))

// Matrix dimensions
const int M = 10000;
const int K = 10000;
const int N = 10000;

// Number of independent matrix multiplications in the batch
const int NUM_MULS    = 10;
const int BLOCK_SIZE  = 16;
const int NUM_STREAMS = NUM_MULS;  // one stream per multiply

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

// Tiled matrix multiplication kernel: C = A * B
__global__
void matmulTiledKernel(const float *A, const float *B, float *C, int m, int k, int n) {
  int TILE = blockDim.x;  // assume blockDim.x == blockDim.y

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

  if (row < m && col < n){
    C[IDX2C(row, col, n)] = sum;
  }
}

int main() {
  srand(67);

  int m = M, k = K, n = N;

  size_t sizeA = (size_t)m * k * sizeof(float);
  size_t sizeB = (size_t)k * n * sizeof(float);
  size_t sizeC = (size_t)m * n * sizeof(float);

  // Host buffers (A,B reused for all multiplies)
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

  // Only do CPU verification for smaller sizes
  bool doVerify = (m <= 512 && k <= 512 && n <= 512);
  if (doVerify) {
    printf("Running CPU reference multiply for verification...\n");
    matmulCPU(h_A, h_B, h_Cref, m, k, n);
  } else {
    printf("Skipping CPU verification for large size (%d x %d * %d x %d)\n", m, k, k, n);
  }

  // Device buffers (one set reused for all multiplies in baseline)
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeC);
  checkCUDAError("cudaMalloc baseline buffers");

  // Kernel configuration
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t sharedBytes = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  // NUM_MULS multiplies, no streams
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int iter = 0; iter < NUM_MULS; ++iter) {
    // H2D copies (synchronous)
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Kernel
    matmulTiledKernel<<<grid, block, sharedBytes>>>(
      d_A, d_B, d_C, m, k, n
    );
    cudaDeviceSynchronize();
    checkCUDAError("matmulTiledKernel (baseline)");

    // D2H copy
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float totalMsBaseline = 0.0f;
  cudaEventElapsedTime(&totalMsBaseline, start, stop);

  // FLOP accounting: each GEMM is 2*m*k*n
  double opsPer   = 2.0 * (double)m * (double)k * (double)n;
  double totalOps = opsPer * (double)NUM_MULS;
  double secs     = totalMsBaseline / 1000.0;
  double gflopsBaseline = totalOps / (secs * 1.0e9);

  printf("[Baseline] %d multiplies, total time: %.3f ms, throughput: %.2f GFLOPs\n", NUM_MULS, totalMsBaseline, gflopsBaseline);

  // Optional correctness check (only for small matrices)
  if (doVerify) {
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    matmulTiledKernel<<<grid, block, sharedBytes>>>(
      d_A, d_B, d_C, m, k, n
    );
    cudaDeviceSynchronize();
    checkCUDAError("verification kernel");

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    float maxDiff = maxAbsDiff(h_C, h_Cref, m * n);
    printf("[Baseline] Max abs diff vs CPU: %e\n", maxDiff);
  }

  // NUM_MULS streams + async copies 

  cudaStream_t streams[NUM_STREAMS];
  float *d_A_stream[NUM_STREAMS];
  float *d_B_stream[NUM_STREAMS];
  float *d_C_stream[NUM_STREAMS];

  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&d_A_stream[i], sizeA);
    cudaMalloc(&d_B_stream[i], sizeB);
    cudaMalloc(&d_C_stream[i], sizeC);
  }
  checkCUDAError("cudaMalloc stream buffers / stream create");

  cudaEvent_t sStart, sStop;
  cudaEventCreate(&sStart);
  cudaEventCreate(&sStop);

  // Schedule all multiplies across NUM_STREAMS streams
  cudaEventRecord(sStart);
  for (int iter = 0; iter < NUM_MULS; ++iter) {
    int s = iter % NUM_STREAMS;

    // Async H2D copies in stream s
    cudaMemcpyAsync(d_A_stream[s], h_A, sizeA, cudaMemcpyHostToDevice, streams[s]);
    cudaMemcpyAsync(d_B_stream[s], h_B, sizeB, cudaMemcpyHostToDevice, streams[s]);

    // Kernel in stream s
    matmulTiledKernel<<<grid, block, sharedBytes, streams[s]>>>(
      d_A_stream[s], d_B_stream[s], d_C_stream[s], m, k, n
    );

    // Async D2H copy in stream s
    cudaMemcpyAsync(h_C, d_C_stream[s], sizeC, cudaMemcpyDeviceToHost, streams[s]);
  }

  // Wait for all streams to finish
  cudaDeviceSynchronize();
  cudaEventRecord(sStop);
  cudaEventSynchronize(sStop);

  float totalMsStreams = 0.0f;
  cudaEventElapsedTime(&totalMsStreams, sStart, sStop);
  checkCUDAError("streamed GEMMs");

  double secsStreams = totalMsStreams / 1000.0;
  double gflopsStreams = totalOps / (secsStreams * 1.0e9);

  printf("[Streams]  %d multiplies, total time: %.3f ms, throughput: %.2f GFLOPs\n", NUM_MULS, totalMsStreams, gflopsStreams);

  // Cleanup streams
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaFree(d_A_stream[i]);
    cudaFree(d_B_stream[i]);
    cudaFree(d_C_stream[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaEventDestroy(sStart);
  cudaEventDestroy(sStop);

  // Cleanup baseline device buffers
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Cleanup host
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Cref);

  return 0;
}