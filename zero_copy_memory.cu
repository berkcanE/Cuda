#include <iostream>
#include <cuda_runtime.h>

void initialData(float *IP, int size)
{
    for(int i = 0 ; i < size ; i++)
    {
        ip[i] = i;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}


__global__ void sumArrays(float *A, float *B, float *C, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    int n_repeat = 3;
    if (idx < N)
    {
        for (int i = 0; i < n_repeat; i++)
        {
        C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, unsigned int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}




int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if(!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d: %s ", dev, deviceProp.name);

    // set up date size of vectors
    int ipower = 10;
    if(argc > 1) ipower = atoi(argv[1]);

    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18) 
    {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem,\
        ipower,(float)nBytes/(1024.0f));
    }
    else
    {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem,\
        ipower,(float)nBytes/(1024.0f*1024.0f));
    }


    // part1: using device memory
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vecotr at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // mallocdevice global memory
    float d_A, d_B, d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    int iLen = 512;
    dim3 block(iLen);
    dim3 grid ((nElem+block.x-1)/block.x);

    sumArrays <<<grid, block>>>(d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);




    // Part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&h_A, nBytes, flags);
    cudaHostAlloc((void **)&h_B, nBytes, flags);

    // initialize data at host size
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    // pass pointer to device
    cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);

    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    sumArraysZeroCopy <<<grid, block>>>(d_A, d_B, d_C, nElem);
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free memory
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    free(hostRef);
    free(gpuRef);
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;

}

// nvcc -O3 -arch=sm_20 sumArrayZerocpy.cu -o sumZerocpy 


// $ nvprof ./sumZerocpy 
// Using Device 0: Tesla M2090 Vector size 1024 power 10 nbytes 4 KB
// Time(%) Time Calls Avg Min Max Name
//  27.18% 3.7760us 1 3.7760us 3.7760us 3.7760us sumArraysZeroCopy
//  11.80% 1.6390us 1 1.6390us 1.6390us 1.6390us sumArrays
//  25.56% 3.5520us 3 1.1840us 1.0240us 1.5040us [CUDA memcpy HtoD]
//  35.47% 4.9280us 2 2.4640us 2.4640us 2.4640us [CUDA memcpy DtoH]