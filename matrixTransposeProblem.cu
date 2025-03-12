#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>


void initialData(float *IP, int size)
{
    for(int i = 0 ; i < size ; i++)
    {
        IP[i] = i;
    }
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; ++iy)
    {
       for (int ix = 0; ix < nx; ++ix)
       {
          out[ix*ny+iy] = in[iy*nx+ix];
       }
    }
}

__global__ void copyRow(float* out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float* out, float* in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

__global__ void transposeNaiveRow(float* out, float* in , const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float* out, float* in , const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

int main(int argc, char** argv)
{
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device: %d: %s\n", device, deviceProp.name);

    // set up array size 2048
    int nx = 1 << 11; // 2048
    int ny = 1 << 11; // 2048

    // select a kernel and a blocksize
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if(argc > 1) iKernel = atoi(argv[1]);
    if(argc > 2) blockx = atoi(argv[2]);
    if(argc > 3) blocky = atoi(argv[3]);
    if(argc > 4) nx = atoi(argv[4]);
    if(argc > 5) ny = atoi(argv[5]);

    printf("With matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float* h_A = (float *)malloc(nBytes);
    float* hostRef = (float *)malloc(nBytes);
    float* gpuRef = (float *)malloc(nBytes);

    // initialize data
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // Kernel execution timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run kernel
    cudaEventRecord(start);
    
    if (iKernel == 0)
    {
        printf("Running CopyRow kernel...\n");
        copyRow<<<grid, block>>>(d_C, d_A, nx, ny);
    }
    else if (iKernel == 1)
    {
        printf("Running CopyCol kernel...\n");
        copyCol<<<grid, block>>>(d_C, d_A, nx, ny);
    }
    else if(iKernel == 2)
    {
        printf("Running transposeNaiveRow kernel...\n");
        transposeNaiveRow<<<grid, block>>>(d_C, d_A, nx, ny);
    }
    else if(iKernel == 3)
    {
        printf("Running transposeNaiveCol kernel...\n");
        transposeNaiveCol<<<grid, block>>>(d_C, d_A, nx, ny);
    }
    else
    {
        printf("Invalid kernel selection!\n");
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate effective bandwidth
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / (milliseconds / 1000.0);
    printf("Elapsed time: %f ms, Effective bandwidth: %f GB/s\n", milliseconds, ibnd);

    // Check kernel results
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // free host and device memory
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
// The NaiveCol approach performs better than the NaiveRow approach. As explained earlier, one  likely cause of this improvement in performance is that the strided reads are cached.
