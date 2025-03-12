#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduceNioghboredLess(int* inputData, int* outputData, unsigned int n)
{
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idata = inputData + blockDim.x * blockIdx.x;

    for(int stride i = 1 ; stride < blockDim.x ; stride *=2)
    {
        int index = 2 * stride * tid;
        if(index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    if(tid == 0) outputData[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int* inputData, int* outputData, unisigned int n)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int* idata = inputData + blockDim.x * threadIdx.x * 2;

    if(idx + blockDim.x < n)
    {
        inputData[idx] += inputData[idx + blockDim.x];
    }

    for(int stride = blockDim.x / 2 ; stride > 0 ; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) outputData[blockIdx.x] = idata[0];
}