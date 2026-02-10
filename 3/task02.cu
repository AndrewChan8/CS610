#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#define IMAGE_DIM 2048

using uchar = unsigned char;

void output_image_file(uchar* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);

// AoS grayscale kernel
__global__ void image_to_grayscale(const uchar3 *image, uchar *image_output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= IMAGE_DIM || y >= IMAGE_DIM) return;

  int idx = y * IMAGE_DIM + x;
  uchar3 p = image[idx];

  float gray = 0.21f * p.x + 0.72f * p.y + 0.07f * p.z;
  image_output[idx] = (uchar)gray;
}

// SoA grayscale kernel
__global__ void image_to_grayscale_soa(
  const uchar *R,
  const uchar *G,
  const uchar *B,
  uchar *image_output
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= IMAGE_DIM || y >= IMAGE_DIM) return;

  int idx = y * IMAGE_DIM + x;

  float gray = 0.21f * R[idx] + 0.72f * G[idx] + 0.07f * B[idx];
  image_output[idx] = (uchar)gray;
}

int main(void) {
  unsigned int image_size, image_output_size;
  uchar3 *h_image, *d_image;
  uchar *h_image_output, *d_image_output;
  uchar *h_R, *h_G, *h_B;
  uchar *d_R, *d_G, *d_B;

  cudaEvent_t start, stop;
  float ms, ms_aos, ms_soa;

  image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar3);
  image_output_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  checkCUDAError("event create");

  cudaMalloc(&d_image, image_size);
  cudaMalloc(&d_image_output, image_output_size);
  checkCUDAError("device malloc");

  h_image = (uchar3*)malloc(image_size);
  h_image_output = (uchar*)malloc(image_output_size);

  input_image_file((char*)"input.ppm", h_image);

  cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
  checkCUDAError("H2D AoS");

  dim3 block(16, 16);
  dim3 grid(
    (IMAGE_DIM + block.x - 1) / block.x,
    (IMAGE_DIM + block.y - 1) / block.y
  );

  cudaEventRecord(start);
  image_to_grayscale<<<grid, block>>>(d_image, d_image_output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCUDAError("AoS kernel");

  cudaEventElapsedTime(&ms, start, stop);
  ms_aos = ms;

  h_R = (uchar*)malloc(image_output_size);
  h_G = (uchar*)malloc(image_output_size);
  h_B = (uchar*)malloc(image_output_size);

  int n = IMAGE_DIM * IMAGE_DIM;
  for (int i = 0; i < n; i++) {
    h_R[i] = h_image[i].x;
    h_G[i] = h_image[i].y;
    h_B[i] = h_image[i].z;
  }

  cudaMalloc(&d_R, image_output_size);
  cudaMalloc(&d_G, image_output_size);
  cudaMalloc(&d_B, image_output_size);

  cudaMemcpy(d_R, h_R, image_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_G, h_G, image_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, image_output_size, cudaMemcpyHostToDevice);
  checkCUDAError("H2D SoA");

  cudaEventRecord(start);
  image_to_grayscale_soa<<<grid, block>>>(d_R, d_G, d_B, d_image_output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCUDAError("SoA kernel");

  cudaEventElapsedTime(&ms, start, stop);
  ms_soa = ms;

  cudaMemcpy(
    h_image_output,
    d_image_output,
    image_output_size,
    cudaMemcpyDeviceToHost
  );
  checkCUDAError("D2H");

  printf("AoS execution time: %f ms\n", ms_aos);
  printf("SoA execution time: %f ms\n", ms_soa);

  output_image_file(h_image_output);

  cudaFree(d_image);
  cudaFree(d_image_output);
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);

  free(h_image);
  free(h_image_output);
  free(h_R);
  free(h_G);
  free(h_B);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

void output_image_file(uchar* image) {
  FILE *f = fopen("output.ppm", "wb");
  if (!f) exit(1);

  fprintf(f, "P5\n%d %d\n255\n", IMAGE_DIM, IMAGE_DIM);

  for (int y = 0; y < IMAGE_DIM; y++) {
    for (int x = 0; x < IMAGE_DIM; x++) {
      int i = x + y * IMAGE_DIM;
      fwrite(&image[i], sizeof(uchar), 1, f);
    }
  }

  fclose(f);
}

void input_image_file(char* filename, uchar3* image) {
  FILE *f = fopen("input.ppm", "rb");
  if (!f) exit(1);

  char tmp[256];
  int x, y, s;
  fscanf(f, "%s\n", tmp);
  fscanf(f, "%d %d\n", &x, &y);
  fscanf(f, "%d\n", &s);

  for (int yi = 0; yi < IMAGE_DIM; yi++) {
    for (int xi = 0; xi < IMAGE_DIM; xi++) {
      int i = xi + yi * IMAGE_DIM;
      fread(&image[i], sizeof(uchar), 3, f);
    }
  }

  fclose(f);
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}