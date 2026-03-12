# Assignment 6 Report - Andrew Chan

This assignment looks at 3 parallel algorithm patterns which are for histogram, prefix scan, and merge. 

## Task 1 

In task 1 a GPU kernel was implemented to compute a histogram over an input array. Each thread processes one innput element and determines the corresponding histogram. 

To reduce contention from global memory, histogram privatization was used. Each thread block maintains a shared memory histogram where threads update bin counts using atomic operations. After all threads complete their updates, the shared memory histogram is merged into the global histogram using atomic additions. 

## Task 2

In task 2 a parallel prefix scan was implemented using the Kogge stone algorithm. The scan computes cumulative sums across the input array where each output element represents the sum of all preceding elements including itself. 

The implementation loads the input array into shared memory and performs iterative stride based updates. During each of the iterations, threads read values from earlier positions and update their partial sums. 

Each thread loads two elements into shared memory to fully utilize abailable parallelism and reduce idle threads.

## Task 3
In task 3 I implemented a GPU based merge of two sorted arrays. The kernel uses a tiled merge where the output array is partitioned into tiles processed by independent thread blocks. 

This technique is used to determine which portions of arrays A and B correspond to each output tile. Each block loads its corresponding segments into shared memory before performing the merge operation in parallel.