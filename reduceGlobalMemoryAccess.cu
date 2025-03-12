#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduceGMem(int * g_idata, int* g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    int idata = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= n) return;


    if(blockDim.x >= 1024 && tid < 512){    idata[tid] += idata[tid + 512];  __syncthreads();   }
    if(blockDim.x >= 512 && tid < 256){     idata[tid] += idata[tid + 256];  __syncthreads();   }
    if(blockDim.x >= 256 && tid < 128){     idata[tid] += idata[tid + 128];  __syncthreads();   }
    if(blockDim.x >= 128 && tid < 64){      idata[tid] == idata[tid + 64];   __syncthreads();   }

    // unrolling warp
    if(tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// reduce at device 0: Tesla K40c with array size 16777216  grid 131072 block 128
// Time(%)      Time     Calls       Avg       Min       Max  Name
//  2.01%  2.1206ms         1  2.1206ms  2.1206ms  2.1206ms  reduceGmem()
//  1.10%  1.1536ms         1  1.1536ms  1.1536ms  1.1536ms  reduceSmem()