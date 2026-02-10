#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#define IMAGE_DIM 2048
#define BLOCK_DIM 16

void output_image_file(uchar3* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);

__global__ void image_blur_A(uchar3 *image, uchar3 *image_output);
__global__ void image_blur_B(uchar3 *image, uchar3 *image_output);
__global__ void image_blur_C(uchar3 *image, uchar3 *image_output);
__global__ void image_blur_D(uchar3 *image, uchar3 *image_output);

__device__ __forceinline__
int wrap_coord(int v) {
  if (v < 0) v += IMAGE_DIM;
  if (v >= IMAGE_DIM) v -= IMAGE_DIM;
  return v;
}

extern __shared__ uchar3 sh_tile[];

template<int R>
__device__ void blur_impl(uchar3 *image, uchar3 *image_output) {
  const int TILE_W = BLOCK_DIM + 2 * R;
  const int TILE_H = BLOCK_DIM + 2 * R;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int x = bx * BLOCK_DIM + tx;
  int y = by * BLOCK_DIM + ty;

  // Load tile with halo into shared memory
  for (int dy = ty; dy < TILE_H; dy += BLOCK_DIM) {
    for (int dx = tx; dx < TILE_W; dx += BLOCK_DIM) {
      int gx = bx * BLOCK_DIM + dx - R;
      int gy = by * BLOCK_DIM + dy - R;

      gx = wrap_coord(gx);
      gy = wrap_coord(gy);

      sh_tile[dy * TILE_W + dx] = image[gy * IMAGE_DIM + gx];
    }
  }
  __syncthreads();

  if (x >= IMAGE_DIM || y >= IMAGE_DIM) {
    return;
  }

  int local_x = tx + R;
  int local_y = ty + R;

  float r_sum = 0.0f;
  float g_sum = 0.0f;
  float b_sum = 0.0f;

  const int DIAM = 2 * R + 1;
  const float weight = 1.0f / (float)(DIAM * DIAM);

  for (int j = -R; j <= R; ++j) {
    int ly = local_y + j;
    for (int i = -R; i <= R; ++i) {
      int lx = local_x + i;
      uchar3 pix = sh_tile[ly * TILE_W + lx];
      r_sum += (float)pix.x;
      g_sum += (float)pix.y;
      b_sum += (float)pix.z;
    }
  }

  uchar3 out;
  out.x = (unsigned char)(r_sum * weight + 0.5f);
  out.y = (unsigned char)(g_sum * weight + 0.5f);
  out.z = (unsigned char)(b_sum * weight + 0.5f);

  image_output[y * IMAGE_DIM + x] = out;
}

// Kernel wrappers for the four radii

__global__ void image_blur_A(uchar3 *image, uchar3 *image_output) {
  blur_impl<1>(image, image_output);
}

__global__ void image_blur_B(uchar3 *image, uchar3 *image_output) {
  blur_impl<2>(image, image_output);
}

__global__ void image_blur_C(uchar3 *image, uchar3 *image_output) {
  blur_impl<4>(image, image_output);
}

__global__ void image_blur_D(uchar3 *image, uchar3 *image_output) {
  blur_impl<8>(image, image_output);
}

/* Host code */

int main(int argc, char **argv) {
  unsigned int image_size;
  uchar3 *d_image, *d_image_output;
  uchar3 *h_image;
  cudaEvent_t start, stop;
  float ms = 0.0f;

  image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar3);

  // Choose radius / kernel: default 1 (A)
  int radius = 1;
  if (argc >= 2) {
    int r = atoi(argv[1]);
    if (r == 1 || r == 2 || r == 4 || r == 8) {
      radius = r;
    }
  }

  // create timers
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on the GPU for the images
  cudaMalloc((void**)&d_image, image_size);
  cudaMalloc((void**)&d_image_output, image_size);
  checkCUDAError("CUDA malloc");

  // allocate and load host image
  h_image = (uchar3*)malloc(image_size);
  if (!h_image) {
    fprintf(stderr, "Host malloc failed\n");
    return EXIT_FAILURE;
  }
  input_image_file((char*)"input.ppm", h_image);

  // copy image to device memory
  cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
  checkCUDAError("CUDA memcpy to device");

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((IMAGE_DIM + BLOCK_DIM - 1) / BLOCK_DIM, (IMAGE_DIM + BLOCK_DIM - 1) / BLOCK_DIM);

  size_t shared_bytes = (BLOCK_DIM + 2 * radius) * (BLOCK_DIM + 2 * radius) * sizeof(uchar3);

  // launch appropriate kernel
  cudaEventRecord(start);
  switch (radius) {
    case 1:
      image_blur_A<<<grid, block, shared_bytes>>>(d_image, d_image_output);
      break;
    case 2:
      image_blur_B<<<grid, block, shared_bytes>>>(d_image, d_image_output);
      break;
    case 4:
      image_blur_C<<<grid, block, shared_bytes>>>(d_image, d_image_output);
      break;
    case 8:
      image_blur_D<<<grid, block, shared_bytes>>>(d_image, d_image_output);
      break;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCUDAError("Kernel launch");

  cudaEventElapsedTime(&ms, start, stop);

  // copy the image back from the GPU for output to file
  cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
  checkCUDAError("CUDA memcpy from device");

  // output timings
  printf("Execution time:\t%f\n", ms);

  // output image
  output_image_file(h_image);

  // cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_image);
  cudaFree(d_image_output);
  free(h_image);

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
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", &x, &y);
  fscanf(f, "%d\n",&s);
  if ((x != y) && (x != IMAGE_DIM)){
    fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
    exit(1);
  }

  for (int xi = 0; xi < IMAGE_DIM; xi++){
    for (int yi = 0; yi < IMAGE_DIM; yi++){
      int i = xi + yi*IMAGE_DIM;
      fread(&image[i], sizeof(unsigned char), 3, f);
    }
  }

  fclose(f);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}