# Assignment 5 - Andrew Chan

This assignment explores GPU memory bandwidth measurement and parallel ray tracing performance. In Task 1 we evaluate  the practical versus theoretical global memory bandwidth using a vector addition kernel. In Task 2, we implement a ray tracer on the GPU and analyze how execution time scales as the number of spheres increases.

In Task 1, a vector addition kernel was implemented using statically allocated device globals instead of cudaMalloc. Each thread computes:
```
c[i] = a[i] + b[i]
```

The kernel execution time was measured using CUDA event. The memory bandwidth was computd using the device memory clock rate and global memory bus width. 

For N = 4,194,304 integers, each element requires two reads and one write. This results in around 48 MB of memory traffic per kernel launch.

Here are the results on the NVIDIA A100 80GB PCIe:
Kernel time: ~0.127 ms
Measured bandwidth: ~397 GB/s
Theoretical bandwidth: ~1935 GB/s

In Task 2, a GPU ray tracer was implemented. One thread was launched per pixel for a 2048x2048 image. Each thread traced a ray though the image plan and tested intersection against all spheres. For each pixel, the closest valid sphere intersection was selected and color intensity was scaled based on distance from the sphere center. 

Since each pixel iterates over every sphere, runtime scales linearly with sphere count and since the number of pixels is fixed, total work is proportional to the number of spheres.

The timing results:
16 spheres: ~0.139 ms
32 spheres: ~0.155 ms
64 spheres: ~0.287 ms
128 spheres: ~0.548 ms
256 spheres: ~1.071 ms
512 spheres: ~2.124 ms
1024 spheres: ~4.212 ms
2048 spheres: ~8.390 ms