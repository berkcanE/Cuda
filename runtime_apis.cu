#include <stdio.h>
#include <cuda_runtime.h>


int main()
{
    int nDevice;
    cudaGetDeviceCount(&nDevice);

    for(int i = 0 ; i < nDevice ; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        int maxThreadsPerMP = 0;
        cudaDeviceGetAttribute(&maxThreadsPerMP, cudaDevAttrMaxThreadsPerMultiProcessor, nDevice);


        printf("Device Number: %d,\n"
        "Device Name: %s,\n"
        "Memory Clock Rate (KHz): %d,\n"
        "Memory Bus Width(bits): %d,\n"
        "Peak Memory Bandwidth(GB/s): %f,\n\n"
        "Total Global Memory: %.2f GB,\n"
        "Compute Capability: %d.%d,\n"
        "Number Of SMs: %d,\n"
        "Max threads per block: %d,\n"
        "Max thread dimension: x= %d, y= %d, z= %d,\n"
        "Max grid dimension: x= %d, y= %d, z= %d,\n"
        "Max_threads_per_SM 0: %d\n"
        "Max_warps_per_SM 0: %d\n"
        "Max threads_per_SM 1: %d   \n"
        "Max_warps_per_SM 1: %d\n",
        i,
        deviceProp.name,
        deviceProp.memoryClockRate,
        deviceProp.memoryBusWidth,
        2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8.0) / 1.0e6,
        static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
        deviceProp.major, deviceProp.minor,
        deviceProp.multiProcessorCount,
        deviceProp.maxThreadsPerBlock,
        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2],
        deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2],
        deviceProp.maxThreadsPerMultiProcessor,
        deviceProp.maxThreadsPerMultiProcessor/32,
        maxThreadsPerMP,
        maxThreadsPerMP/32);


        
    }

    return 0;
}