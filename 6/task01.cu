#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define HISTO_SIZE 256

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void generate_data(char * data, unsigned int length) {
	for (unsigned int i = 0; i < length; ++i) {
		data[i] = (char)(rand() % HISTO_SIZE);
	}
}

__global__ void GPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	__shared__ unsigned int local_histo[HISTO_SIZE];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize shared histogram
	if (tid < HISTO_SIZE) {
		local_histo[tid] = 0;
	}
	__syncthreads();

	// Each thread processes one input element
	if (gid < length) {
		unsigned int bin = (unsigned char)data[gid];
		atomicAdd(&local_histo[bin], 1);
	}
	__syncthreads();

	// Merge shared histogram into global histogram
	if (tid < HISTO_SIZE) {
		atomicAdd(&histo[tid], local_histo[tid]);
	}
}

void CPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	for (unsigned int i = 0; i < HISTO_SIZE; i++) {
		histo[i] = 0;
	}

	for (unsigned int i = 0; i < length; i++) {
		unsigned int bin = (unsigned char)data[i];
		histo[bin]++;
	}
}

/* Host code */
int main(void) {
	unsigned int input_length = 2048;
	char * h_data, * d_data;
	unsigned int * h_histo, * d_histo;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size = input_length * sizeof(char);
	unsigned int histo_size = HISTO_SIZE * sizeof(unsigned int);

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU
	cudaMalloc((void**)&d_data, data_size);
	cudaMalloc((void**)&d_histo, histo_size);
	cudaMemset(d_histo, 0, histo_size);
	checkCUDAError("CUDA malloc / memset");

	// allocate host data
	h_data = (char*)malloc(data_size);
	h_histo = (unsigned int*)malloc(histo_size);
	generate_data(h_data, input_length);

	// copy input to device memory
	cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	int blockSize = 256;
	int gridSize = (input_length + blockSize - 1) / blockSize;

	cudaEventRecord(start, 0);

	GPU_histogram<<<gridSize, blockSize>>>(d_data, input_length, d_histo);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel launch");

	// copy the histogram back from the GPU
	cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// output timings
	printf("Execution time:\t%f ms\n", ms);

	// Verify output using CPU
	unsigned int *cpu_histo = (unsigned int*)malloc(histo_size);
	CPU_histogram(h_data, input_length, cpu_histo);

	int correct = 1;
	for (unsigned int i = 0; i < HISTO_SIZE; i++) {
		if (h_histo[i] != cpu_histo[i]) {
			printf("Mismatch at bin %u: GPU=%u CPU=%u\n", i, h_histo[i], cpu_histo[i]);
			correct = 0;
			break;
		}
	}

	if (correct) {
		printf("Histogram verification: PASSED\n");
	} else {
		printf("Histogram verification: FAILED\n");
	}

	// cleanup
	free(cpu_histo);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_data);
	cudaFree(d_histo);
	free(h_data);
	free(h_histo);

	return 0;
}