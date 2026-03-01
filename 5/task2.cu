#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>


#define IMAGE_DIM 2048

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void output_image_file(uchar3* image, std::string filename);
void checkCUDAError(const char *msg);

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
};

/* Device Code */

__constant__ unsigned int d_sphere_count;

__global__ void ray_trace(uchar3 *image, Sphere *d_s) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Add your implementation here


	image[offset].x = (int)(r);
	image[offset].y = (int)(g);
	image[offset].z = (int)(b);
}

/* Host code */

float test(unsigned int sphere_count) {
	unsigned int image_size, spheres_size;
	uchar3 *d_image;
	uchar3 *h_image;
	cudaEvent_t     start, stop;
	Sphere h_s[sphere_count];
	Sphere *d_s;
	float timing_data;

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar3);
	spheres_size = sizeof(Sphere)*sphere_count;

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_s, spheres_size);
	checkCUDAError("CUDA malloc");

	// create some random spheres
	for (int i = 0; i<sphere_count; i++) {
		h_s[i].r = rnd(1.0f)*255;
		h_s[i].g = rnd(1.0f)*255;
		h_s[i].b = rnd(1.0f)*255;
		h_s[i].x = rnd((float)IMAGE_DIM);
		h_s[i].y = rnd((float)IMAGE_DIM);
		h_s[i].z = rnd((float)IMAGE_DIM);
		h_s[i].radius = rnd(100.0f) + 20;
	}
	//copy to device memory
	cudaMemcpy(d_s, h_s, spheres_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	//generate host image
	h_image = (uchar3*)malloc(image_size);

	//cuda layout
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(unsigned int));
	checkCUDAError("CUDA copy sphere count to device");

	// generate a image from the sphere data
	cudaEventRecord(start, 0);
	ray_trace << <blocksPerGrid, threadsPerBlock >> >(d_image, d_s);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timing_data, start, stop);
	checkCUDAError("kernel (normal)");


	// copy the image back from the GPU for output to file
	cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// output image
	output_image_file(h_image, "output_" + std::to_string(sphere_count) + ".ppm");

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_s);
	free(h_image);

	return timing_data;
}

void output_image_file(uchar3* image, std::string filename)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen(filename.c_str(), "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
		}
	}
	
	fclose(f);
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

int main() {

	printf("Timing Data Table\n Spheres | Time\n");
	for (unsigned int sphere_count = 16; sphere_count <= 2048; sphere_count *= 2) {
		float timing_data = test(sphere_count);
		printf(" %-7i | %-6.3f\n", sphere_count, timing_data);
	}
}
