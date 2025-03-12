#include <iostream>
#include <cuda_runtime.h>

// 1. Partition the input vector into smaller chunks.
// 2. Have a thread calculate the partial sum for each chunk.
// 3. Add the partial results from each chunk into a final sum.
int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

    cudaSetDevice(dev);

    int size = 1 << 24;
    printf(" with array size %d\n", size);

    int minGridSize, bestBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, (void *)reduceNeighbored, 0, 0);
    printf("Recommended block size: %d\n", bestBlockSize);

    int blockSize = (argc > 1) ? atoi(argv[1]) : bestBlockSize;

    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // Allocate host memory
    size_t nBytes = size * sizeof(int);
    int *h_idata = (int *)malloc(nBytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(nBytes);

    // Initialize input data
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, nBytes);

    // Allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, nBytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute GPU sum
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("GPU Neighbored elapsed time: %.3f ms, gpu_sum: %d <<<grid %d block %d>>>\n",
           milliseconds, gpu_sum, grid.x, block.x);

    // Compute CPU sum
    int cpu_sum = recursiveReduce(tmp, size);

    // Free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // Free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    // Reset device
    cudaDeviceReset();

    // Check results
    if (gpu_sum != cpu_sum)
    {
        printf("Test failed! GPU sum: %d, CPU sum: %d\n", gpu_sum, cpu_sum);
        return EXIT_FAILURE;
    }
    else
    {
        printf("Test passed!\n");
    }

    return EXIT_SUCCESS;
}
