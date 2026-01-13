#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Task 03.5: non-square matrix dimensions
#define ROWS 2048
#define COLS 2050

#define TILE 16  // 16x16 threads per block

void checkCUDAError(const char*);
void random_ints(int *a, int len);

void matrixAddCPU(const int *a, const int *b, int *c_ref, int width, int height);
int validate(const int *c, const int *c_ref, int len);

__global__ void matrixAdd(const int *a, const int *b, int *c, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x; // x -> columns
	int row = blockIdx.y * blockDim.y + threadIdx.y; // y -> rows

	if (row < height && col < width) {
		int idx = row * width + col;  // row-major flatten
		c[idx] = a[idx] + b[idx];
	}
}

int main(void)
{
	int *a, *b, *c, *c_ref;     // host
	int *d_a, *d_b, *d_c;       // device

	const int width  = COLS;
	const int height = ROWS;

	const size_t numElems = (size_t)width * (size_t)height;
	const size_t size = numElems * sizeof(int);

	// Device alloc
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Host alloc + init
	a     = (int*)malloc(size);
	b     = (int*)malloc(size);
	c     = (int*)malloc(size);
	c_ref = (int*)malloc(size);

	random_ints(a, (int)numElems);
	random_ints(b, (int)numElems);

	// CPU reference
	matrixAddCPU(a, b, c_ref, width, height);

	// H->D
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy H2D");

	// 2D launch
	dim3 threadsPerBlock(TILE, TILE);
	dim3 blocksPerGrid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

	matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);

	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");

	// D->H
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy D2H");

	// Validate
	int errors = validate(c, c_ref, (int)numElems);
	printf("Total errors: %d\n", errors);

	// Cleanup
	free(a); free(b); free(c); free(c_ref);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a, int len)
{
	for (int i = 0; i < len; i++) a[i] = rand();
}

void matrixAddCPU(const int *a, const int *b, int *c_ref, int width, int height)
{
	int len = width * height;
	for (int i = 0; i < len; i++) c_ref[i] = a[i] + b[i];
}

int validate(const int *c, const int *c_ref, int len)
{
	int errors = 0;
	for (int i = 0; i < len; i++) {
		if (c[i] != c_ref[i]) {
			printf("Error at i=%d: GPU=%d CPU=%d\n", i, c[i], c_ref[i]);
			errors++;
		}
	}
	return errors;
}
