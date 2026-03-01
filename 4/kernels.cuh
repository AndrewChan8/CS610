#ifndef KERNEL_H
#define KERNEL_H

// Dataset Switch
// 0 = small (2048 records)
// 1 = large (2^20 records)
#define USE_LARGE_DATA 1

#if USE_LARGE_DATA
    #define NUM_RECORDS (1 << 20)      // 2^20
    #define DATA_FILE   "Student_large.dat"
#else
    #define NUM_RECORDS 2048
    #define DATA_FILE   "Student.dat"
#endif

#define THREADS_PER_BLOCK 256

struct student_record{
    int   student_id;
    float assignment_mark;
};

struct student_records{
    int   student_ids[NUM_RECORDS];
    float assignment_marks[NUM_RECORDS];
};

typedef struct student_record  student_record;
typedef struct student_records student_records;

// ====================
// Task 1 Globals
// ====================
__device__ float d_max_mark             = 0.0f;
__device__ int   d_max_mark_student_id  = 0;
__device__ int   d_lock                 = 0;

// ====================
// Task 1: Atomic Reduction
// ====================
__global__ void maximumMark_atomic_kernel(student_records *d_records){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_RECORDS) return;

    float mark = d_records->assignment_marks[idx];
    int   id   = d_records->student_ids[idx];

    bool done = false;
    while (!done){
        if (atomicCAS(&d_lock, 0, 1) == 0)
        {
            if (mark > d_max_mark)
            {
                d_max_mark = mark;
                d_max_mark_student_id = id;
            }
            __threadfence();
            d_lock = 0;
            done = true;
        }
    }
}

// ====================
// Task 2: Recursive Reduction
// ====================
__global__ void maximumMark_recursive_kernel(student_records *d_in,
                                             student_records *d_out,
                                             int num_records)
{
    extern __shared__ student_record sh_rec[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < num_records) {
        sh_rec[tid].student_id      = d_in->student_ids[idx];
        sh_rec[tid].assignment_mark = d_in->assignment_marks[idx];
    } else {
        sh_rec[tid].student_id      = -1;
        sh_rec[tid].assignment_mark = -1.0f;
    }

    __syncthreads();

    if ((tid % 2 == 0) && (idx + 1 < num_records)) {
        student_record a = sh_rec[tid];
        student_record b = sh_rec[tid + 1];
        student_record best = (b.assignment_mark > a.assignment_mark) ? b : a;

        int out_idx = idx / 2;
        d_out->student_ids[out_idx]      = best.student_id;
        d_out->assignment_marks[out_idx] = best.assignment_mark;
    }
}

// ====================
// Task 3: Shared Memory Reduction
// ====================
__global__ void maximumMark_SM_kernel(student_records *d_records,
                                      student_records *d_reduced_records)
{
    extern __shared__ student_record sh_rec[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < NUM_RECORDS) {
        sh_rec[tid].student_id      = d_records->student_ids[idx];
        sh_rec[tid].assignment_mark = d_records->assignment_marks[idx];
    } else {
        sh_rec[tid].student_id      = -1;
        sh_rec[tid].assignment_mark = -1.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sh_rec[tid + stride].assignment_mark > sh_rec[tid].assignment_mark) {
                sh_rec[tid] = sh_rec[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_reduced_records->student_ids[blockIdx.x]      = sh_rec[0].student_id;
        d_reduced_records->assignment_marks[blockIdx.x] = sh_rec[0].assignment_mark;
    }
}

// ====================
// Task 4: Warp Shuffle Reduction
// ====================
__global__ void maximumMark_shuffle_kernel(student_records *d_records,
                                           student_records *d_reduced_records)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int lane          = threadIdx.x & 31;
    int warp_global   = global_thread >> 5;   // /32

    float max_mark = -1.0f;
    int   max_id   = -1;

    if (global_thread < NUM_RECORDS) {
        max_mark = d_records->assignment_marks[global_thread];
        max_id   = d_records->student_ids[global_thread];
    }

    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_mark = __shfl_down_sync(mask, max_mark, offset);
        int   other_id   = __shfl_down_sync(mask, max_id,   offset);

        if (other_mark > max_mark) {
            max_mark = other_mark;
            max_id   = other_id;
        }
    }

    if (lane == 0) {
        d_reduced_records->assignment_marks[warp_global] = max_mark;
        d_reduced_records->student_ids[warp_global]      = max_id;
    }
}

#endif