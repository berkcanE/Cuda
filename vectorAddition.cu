#include <iostream>
#include <stdio.h>

#define SIZE 1024
#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 1024

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

    // Launch kernel with 1 block of 1024 threads
    vectorAdd<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, SIZE);

    // Copy results from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < SIZE; i++)
    {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
