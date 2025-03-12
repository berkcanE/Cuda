#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define DIMX 32
#define DIMY 32
#define SIZE (DIMX * DIMY)

__global__ void setRowLoadRow(int* out)
{
    __shared__ int tile[DIMY][DIMX];
    
    // row-major ordering when flattening a 2D index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColLoadCol(int* out)
{
    __shared__ int tile[DIMX][DIMY];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowLoadCol(int* out)
{
    __shared__ int tile[DIMY][DIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColLoadRow(int *out)
{
    __shared__ int tile[DIMX][DIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadColDyn(int* out)
{
    extern __shared__ int tile[];

    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    // shared memory store operation
    tile[row_idx] = row_idx;

    __syncthreads();

    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int* out)
{
    __shared__ int tile[DIMY][DIMX + 1];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int* out)
{
    extern __shared__ int tile[];

    unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.y + 1) + threadIdx.y; // Fix: Added missing semicolon

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[row_idx] = g_idx;

    __syncthreads();

    out[g_idx] = tile[col_idx];
}

void runKernel(void (*kernel)(int*), int* d_out, dim3 gridSize, dim3 blockSize, const char* kernelName, size_t sharedMemSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << kernelName << " execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int* d_out;
    cudaMalloc((void**)&d_out, SIZE * sizeof(int));

    // Kernel configurations
    dim3 blockSize(DIMX, DIMY);
    dim3 gridSize(1, 1);

    // Run kernels and compare execution times
    runKernel(setRowLoadRow, d_out, gridSize, blockSize, "setRowLoadRow", 0);
    runKernel(setColLoadCol, d_out, gridSize, blockSize, "setColLoadCol", 0);
    runKernel(setRowLoadCol, d_out, gridSize, blockSize, "setRowLoadCol", 0);
    runKernel(setColLoadRow, d_out, gridSize, blockSize, "setColLoadRow", 0);
    runKernel(setRowReadColDyn, d_out, gridSize, blockSize, "setRowReadColDyn", SIZE * sizeof(int)); // Shared memory size for dynamic
    runKernel(setRowReadColPad, d_out, gridSize, blockSize, "setRowReadColPad", 0);
    runKernel(setRowReadColDynPad, d_out, gridSize, blockSize, "setRowReadColDynPad", (DIMY * (DIMX + 1)) * sizeof(int)); // Shared memory size for padded

    cudaFree(d_out);
    cudaDeviceReset();

    return 0;
}
