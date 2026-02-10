#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>


#define IMAGE_DIM 2048

void output_image_file(uchar3* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);


__global__ void image_blur_A(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	
}

__global__ void image_blur_B(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	
}

__global__ void image_blur_C(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	
}

__global__ void image_blur_D(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	
}


/* Host code */

int main(void) {
	unsigned int image_size;
	uchar3 *d_image, *d_image_output;
	uchar3 *h_image;
	cudaEvent_t start, stop;
	float ms;

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar3);

	// create timers


	// allocate memory on the GPU for the output image

	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar3*)malloc(image_size);
	input_image_file("input.ppm", h_image);

	// copy image to device memory

	checkCUDAError("CUDA memcpy to device");

	// launch kernel

	checkCUDAError("Kernel launch");


	// copy the image back from the GPU for output to file

	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution time:");
	printf("\t%f\n", ms);

	// output image
	output_image_file(h_image);

	//cleanup

	return 0;
}

void output_image_file(uchar3* image)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
}

void input_image_file(char* filename, uchar3* image)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int x, y, s;

	//open the input file and write header info for PPM filetype
	f = fopen("input.ppm", "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'input.ppm' input file\n");
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d %d\n", &x, &y);
	fscanf(f, "%d\n",&s);
	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		exit(1);
	}

	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fread(&image[i], sizeof(unsigned char), 3, f);
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
