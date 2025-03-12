#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);

    printf("Device &d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iPropmultiProcessorCount);
    printf("Total amount of constant memory per block: %4,2f KB\n", iProp.sharedmemPerBlock/1024.0);
    printf("Total numnber of registers available per block: %d\n", iProp.regsPerBlock);
    printf("Warp size%d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor/32);
    return EXIT_SUCCESS
}