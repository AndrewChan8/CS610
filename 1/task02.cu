#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

void vectorAddCPU(int *a, int *b, int *c_ref, int max);
int validate(int *c, int *c_ref, int max);

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < max) {
		c[i] = a[i] + b[i];
	}
}


int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	vectorAddCPU(a, b, c_ref, N);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	errors = validate(c, c_ref, N);
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
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}

void vectorAddCPU(int *a, int *b, int *c_ref, int max)
{
	for (int i = 0; i < max; i++) {
		c_ref[i] = a[i] + b[i];
	}
}

int validate(int *c, int *c_ref, int max)
{
	int errors = 0;
	for (int i = 0; i < max; i++) {
		if (c[i] != c_ref[i]) {
			printf("Error at i=%d: GPU=%d CPU=%d\n", i, c[i], c_ref[i]);
			errors++;
		}
	}
	return errors;
}
