#include <iostream>
#include <stdio.h>

<<<<<<< HEAD
#define SIZE 1024
#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 1024
=======
#define SIZE 1024*1024
// #define NUM_BLOCKS 1
// #define THREADS_PER_BLOCK 1024
>>>>>>> 4d275a7 (cuda)

__global__ void vectorAdd(int *A, int *B, int *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size = SIZE * sizeof(int);  // Size of each array in bytes
<<<<<<< HEAD
=======
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

>>>>>>> 4d275a7 (cuda)

    // Allocate memory on the host
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Initialize arrays A and B
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = i;
        B[i] = SIZE - i;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

<<<<<<< HEAD
    // Launch kernel with 1 block of 1024 threads
    vectorAdd<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, SIZE);

=======

    int threadsPerBlock = 256;
    int blocksPerGrid = (threadsPerBlock  + SIZE - 1) / threadsPerBlock;
    printf("threadsPerBlock: %d, blocksPerGrid: %d", threadsPerBlock, blocksPerGrid);
    cudaEventRecord(start);
    // Launch kernel with 1 block of 1024 threads
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Taken: %.2f", milliseconds);
>>>>>>> 4d275a7 (cuda)
    // Copy results from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
<<<<<<< HEAD
    for (int i = 0; i < SIZE; i++)
    {
        printf("C[%d] = %d\n", i, C[i]);
    }
=======
    // for (int i = 0; i < SIZE; i++)
    // {
    //     printf("C[%d] = %d\n", i, C[i]);
    // }

    printf("C[%d] = %d\n", SIZE, C[SIZE-1]);
>>>>>>> 4d275a7 (cuda)

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
<<<<<<< HEAD
=======
    cudaDeviceReset();
>>>>>>> 4d275a7 (cuda)

    return 0;
}
