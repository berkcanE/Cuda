#include <iostream>

<<<<<<< HEAD
// Use templates 
=======
// Use templates
>>>>>>> 4d275a7 (cuda)

// if you wanna return value from gpu to cpu, use memcpy
__global__ void reduce_in_place(float* input, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    input[index] += input[index + 1];
}


__global__ void reduce_in_place_total_of_two_indeces(float* input, int n)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid % 2 == 0 && index + 1 < n)
    {
        input[index] += input[index + 1];
    }
}

__global__ void imprroved_reduce_in_place_total_of_two_indeces(float* input, int n)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int stride = 1 ; stride < blockDim.x ; stride *= 2)
    {
        __syncthreads();
        if(tid % (2 * stride) == 0 && index + stride < n)
        {
            input[index] += input[index + stride];
        }
    }

    // Store the blokc's reduced result in the first position of this block's
    if(tid == 0)
    {
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
    }
}


float cpu_reduce(float* input, int n)
{
    float sum = 0.0;

    for(int i = 0 ; i < n ; i++)
    {
        sum += input[i];
    }
    return sum;
}


int main()
{
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);
    // allocate a space for the input vector on cpu and gpu
    float* d_input;
    float* h_input = new float[n];
    cudaMalloc(&d_input, size);

    // initialize the input vector on the host size
    for(int i = 0 ; i < n ; i++)
    {
        h_input[i] = static_cast<float>(i);
    }



    // copy host data from the host to the device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // edit the kernel configuration(block size and the grid size)
    int blocksize = 256;
    int gridSize = (n + blocksize - 1) / blocksize;

    float sum = cpu_reduce(h_input, n);
    std::cout <<"CPU output" sum << std::endl;

    namespace normal
    {
    // launch the gpu kernel
    imprroved_reduce_in_place_total_of_two_indeces<<<gridSize, blockSize>>>(d_input, n);
    cudaDeviceSynchronize();

    // gridSize of the first kernel 4096 == partial sums;
    // output of the first kernel = vector(sizeof 4096) - blockSize = 256, so (4096/256) 16 blocks
    imprroved_reduce_in_place_total_of_two_indeces<<<16, blockSize>>>(d_input, 4096);
    cudaDeviceSynchronize();

    // output = number of block (which is 16)
    // a new vector called d_input with a size of 16 elements
    // 1 need one block size because we have 16 elements and we have block size 256
    imprroved_reduce_in_place_total_of_two_indeces<<<1, blockSize>>>(d_input, 16);
    cudaDeviceSynchronize();
    }

    namespace betterSol
    {
        while(gridSize > 1)
        {
            imprroved_reduce_in_place_total_of_two_indeces<<<gridSize, blockSize>>>(d_input, n);
            cudaDeviceSynchronize();

            n = gridSize; // n = 4096, n = 16
            gridSize = (n + blocksize - 1) / blockSize; // (4096 + 256 - 1) / 256 = 16, (16 + 256-1)/256 = 1
        }

        // just the final step
        imprroved_reduce_in_place_total_of_two_indeces<<<1, blockSize>>>(d_input, 16);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&h_input[0], d_input, sizeof(float), cudaMemcpyDeviceToHost);
     
    std::cout << "GPU Output" << h_input[0] << std::endl;

    cudaFree(d_input);
    delete [] h_input;
    return 0;
}