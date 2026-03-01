#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "kernels.cuh"

// Prototypes
void checkCUDAError(const char*);
void readRecords(student_record *records);
void studentRecordAOS2SOA(student_record *aos, student_records *soa);
void maximumMark_cpu(student_records*, float*, int*);
void maximumMark_atomic   (student_records*, student_records*, student_records*, student_records*);
void maximumMark_recursive(student_records*, student_records*, student_records*, student_records*);
void maximumMark_SM       (student_records*, student_records*, student_records*, student_records*);
void maximumMark_shuffle  (student_records*, student_records*, student_records*, student_records*);

int main(void)
{
    student_record   *recordsAOS;
    student_records  *h_records;
    student_records  *h_records_result;
    student_records  *d_records;
    student_records  *d_records_result;

    // Host allocation
    recordsAOS        = (student_record*)malloc(sizeof(student_record) * NUM_RECORDS);
    h_records         = (student_records*)malloc(sizeof(student_records));
    h_records_result  = (student_records*)malloc(sizeof(student_records));

    // Device allocation
    cudaMalloc((void**)&d_records,        sizeof(student_records));
    cudaMalloc((void**)&d_records_result, sizeof(student_records));
    checkCUDAError("CUDA malloc");

    // Read AoS from file and convert to SoA
    readRecords(recordsAOS);
    studentRecordAOS2SOA(recordsAOS, h_records);
    free(recordsAOS);

    // Task 1: atomic CAS
    maximumMark_atomic(h_records, h_records_result, d_records, d_records_result);

    // Task 2: recursive reduction
    maximumMark_recursive(h_records, h_records_result, d_records, d_records_result);

    // Task 3: shared-memory block reduction
    maximumMark_SM(h_records, h_records_result, d_records, d_records_result);

    // Task 4: warp-shuffle reduction
    maximumMark_shuffle(h_records, h_records_result, d_records, d_records_result);

    // Cleanup
    free(h_records);
    free(h_records_result);
    cudaFree(d_records);
    cudaFree(d_records_result);
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

void readRecords(student_record *records)
{
    FILE *f = fopen(DATA_FILE, "rb");   // <-- use DATA_FILE, not hardcoded
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", DATA_FILE);
        exit(1);
    }

    if (fread(records, sizeof(student_record), NUM_RECORDS, f) != NUM_RECORDS) {
        fprintf(stderr, "Error: Unexpected end of file in %s\n", DATA_FILE);
        exit(1);
    }

    fclose(f);
}

// Task 0: AoS -> SoA
void studentRecordAOS2SOA(student_record *aos, student_records *soa)
{
    for (int i = 0; i < NUM_RECORDS; ++i) {
        soa->student_ids[i]      = aos[i].student_id;
        soa->assignment_marks[i] = aos[i].assignment_mark;
    }
}

// CPU Reference
void maximumMark_cpu(student_records *h_records, float *max_mark, int *max_mark_student_id)
{
    float local_max = -1.0f;
    int   local_id  = -1;

    for (int i = 0; i < NUM_RECORDS; ++i) {
        float mark = h_records->assignment_marks[i];
        int   id   = h_records->student_ids[i];

        if (mark > local_max) {
            local_max = mark;
            local_id  = id;
        }
    }

    *max_mark = local_max;
    *max_mark_student_id = local_id;
}

// Task 1: AtomicCAS
void maximumMark_atomic(student_records *h_records, student_records *h_records_result,
                        student_records *d_records, student_records *d_records_result){
    float max_mark = 0.0f;
    int   max_mark_student_id = 0;
    float time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy input to device
    cudaMemcpy(d_records, h_records, sizeof(student_records),
               cudaMemcpyHostToDevice);
    checkCUDAError("Atomics: memcpy H->D");

    // Reset device globals
    float zero_f = 0.0f;
    int   zero_i = 0;
    cudaMemcpyToSymbol(d_max_mark, &zero_f, sizeof(float));
    cudaMemcpyToSymbol(d_max_mark_student_id, &zero_i, sizeof(int));
    cudaMemcpyToSymbol(d_lock, &zero_i, sizeof(int));

    cudaEventRecord(start, 0);

    unsigned int threads_per_block = THREADS_PER_BLOCK;
    unsigned int blocks_per_grid =
        (NUM_RECORDS + threads_per_block - 1) / threads_per_block;

    maximumMark_atomic_kernel<<<blocks_per_grid, threads_per_block>>>(d_records);
    checkCUDAError("Atomics: kernel launch");
    cudaDeviceSynchronize();
    checkCUDAError("Atomics: kernel sync");

    cudaMemcpyFromSymbol(&max_mark, d_max_mark, sizeof(float));
    cudaMemcpyFromSymbol(&max_mark_student_id, d_max_mark_student_id,sizeof(int));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // Validate with CPU
    float cpu_max;
    int   cpu_id;
    maximumMark_cpu(h_records, &cpu_max, &cpu_id);

    if (fabs(cpu_max - max_mark) > 1e-5f) {
        // True mismatch in maximum value
        printf("Atomics: MISMATCH in max value! CPU max=%f (id=%d), GPU max=%f (id=%d)\n",
            cpu_max, cpu_id, max_mark, max_mark_student_id);
    } else if (cpu_id != max_mark_student_id) {
        // Same max mark, different student with same mark – acceptable
        printf("Atomics: NOTE – multiple students with max=%f, CPU picked id=%d, GPU picked id=%d\n",
            max_mark, cpu_id, max_mark_student_id);
    }

    printf("Atomics: Highest mark %f by student %d\n", max_mark, max_mark_student_id);
    printf("\tExecution time %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Task 2: Recursive Reduction
void maximumMark_recursive(student_records *h_records, student_records *h_records_result,
                           student_records *d_records, student_records *d_records_result){
    float max_mark = 0.0f;
    int   max_mark_student_id = 0;
    float time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
    checkCUDAError("Recursive: memcpy H->D");

    int N = NUM_RECORDS;
    student_records *d_in  = d_records;
    student_records *d_out = d_records_result;

    cudaEventRecord(start, 0);

    while (N > THREADS_PER_BLOCK) {
        int threads = THREADS_PER_BLOCK;
        int blocks  = N / threads;    // N is a power-of-two multiple of 256

        size_t shmem_bytes = threads * sizeof(student_record);

        maximumMark_recursive_kernel<<<blocks, threads, shmem_bytes>>>(d_in, d_out, N);
        checkCUDAError("Recursive: kernel launch");
        cudaDeviceSynchronize();
        checkCUDAError("Recursive: kernel sync");

        student_records *tmp = d_in;
        d_in  = d_out;
        d_out = tmp;

        N /= 2;
    }

    cudaMemcpy(h_records_result, d_in, sizeof(student_records), cudaMemcpyDeviceToHost);
    checkCUDAError("Recursive: memcpy D->H");

    max_mark = -1.0f;
    max_mark_student_id = -1;
    for (int i = 0; i < THREADS_PER_BLOCK; ++i) {
        float mark = h_records_result->assignment_marks[i];
        int   id   = h_records_result->student_ids[i];
        if (mark > max_mark) {
            max_mark = mark;
            max_mark_student_id = id;
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    float cpu_max;
    int   cpu_id;
    maximumMark_cpu(h_records, &cpu_max, &cpu_id);
    if (fabs(cpu_max - max_mark) > 1e-5f || cpu_id != max_mark_student_id) {
        printf("Recursive: MISMATCH CPU max=%f (id=%d), GPU max=%f (id=%d)\n",
               cpu_max, cpu_id, max_mark, max_mark_student_id);
    }

    printf("Recursive: Highest mark %f by student %d\n",
           max_mark, max_mark_student_id);
    printf("\tExecution time %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Task 3: Shared-memory block reduction
void maximumMark_SM(student_records *h_records, student_records *h_records_result,
                    student_records *d_records, student_records *d_records_result){
    float max_mark = 0.0f;
    int   max_mark_student_id = 0;
    float time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
    checkCUDAError("SM: memcpy H->D");

    unsigned int threads_per_block = THREADS_PER_BLOCK;
    unsigned int blocks_per_grid = (NUM_RECORDS + threads_per_block - 1) / threads_per_block;

    size_t shmem_bytes = threads_per_block * sizeof(student_record);

    cudaEventRecord(start, 0);

    maximumMark_SM_kernel<<<blocks_per_grid, threads_per_block, shmem_bytes>>>(d_records, d_records_result);
    checkCUDAError("SM: kernel launch");
    cudaDeviceSynchronize();
    checkCUDAError("SM: kernel sync");

    cudaMemcpy(h_records_result, d_records_result, sizeof(student_records),
               cudaMemcpyDeviceToHost);
    checkCUDAError("SM: memcpy D->H");

    max_mark = -1.0f;
    max_mark_student_id = -1;
    for (unsigned int i = 0; i < blocks_per_grid; ++i) {
        float mark = h_records_result->assignment_marks[i];
        int   id   = h_records_result->student_ids[i];
        if (mark > max_mark) {
            max_mark = mark;
            max_mark_student_id = id;
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    float cpu_max;
    int   cpu_id;
    maximumMark_cpu(h_records, &cpu_max, &cpu_id);
    if (fabs(cpu_max - max_mark) > 1e-5f || cpu_id != max_mark_student_id) {
        printf("SM: MISMATCH CPU max=%f (id=%d), GPU max=%f (id=%d)\n",
               cpu_max, cpu_id, max_mark, max_mark_student_id);
    }

    printf("SM: Highest mark %f by student %d\n", max_mark, max_mark_student_id);
    printf("\tExecution time %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Task 4: Warp-shuffle reduction
void maximumMark_shuffle(student_records *h_records, student_records *h_records_result,
                         student_records *d_records, student_records *d_records_result)
{
    float max_mark = 0.0f;
    int   max_mark_student_id = 0;
    float time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
    checkCUDAError("Shuffle: memcpy H->D");

    unsigned int threads_per_block = THREADS_PER_BLOCK;
    unsigned int blocks_per_grid   = (NUM_RECORDS + threads_per_block - 1) / threads_per_block;
    unsigned int warps_per_block   = threads_per_block / 32;
    unsigned int warps_per_grid    = blocks_per_grid * warps_per_block;

    cudaEventRecord(start, 0);

    maximumMark_shuffle_kernel<<<blocks_per_grid, threads_per_block>>>(d_records, d_records_result);
    checkCUDAError("Shuffle: kernel launch");
    cudaDeviceSynchronize();
    checkCUDAError("Shuffle: kernel sync");

    cudaMemcpy(h_records_result, d_records_result, sizeof(student_records),
               cudaMemcpyDeviceToHost);
    checkCUDAError("Shuffle: memcpy D->H");

    max_mark = -1.0f;
    max_mark_student_id = -1;
    for (unsigned int i = 0; i < warps_per_grid; ++i) {
        float mark = h_records_result->assignment_marks[i];
        int   id   = h_records_result->student_ids[i];
        if (mark > max_mark) {
            max_mark = mark;
            max_mark_student_id = id;
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    float cpu_max;
    int   cpu_id;
    maximumMark_cpu(h_records, &cpu_max, &cpu_id);
    if (fabs(cpu_max - max_mark) > 1e-5f || cpu_id != max_mark_student_id) {
        printf("Shuffle: MISMATCH CPU max=%f (id=%d), GPU max=%f (id=%d)\n",
               cpu_max, cpu_id, max_mark, max_mark_student_id);
    }

    printf("Shuffle: Highest mark %f by student %d\n", max_mark, max_mark_student_id);
    printf("\tExecution time %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}