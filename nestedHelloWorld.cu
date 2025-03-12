#include <iostream>
#include <cuda_runtime.h>




__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion: %d, Hellow world from thread: %d block %\n", iDepth, tid, blockIdx.x);

    if(iSize == 1) return;

    int nThreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nThreads > 0)
    {
        nestedHelloWorld<<<1, nThreads>>>(nThreads, ++iDepth);
        printf("--------> nested execution depth: %d\n", iDepth);
    }
}

// nvcc -arch=sm_35 -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt


// ./nestedHelloWorld Execution Configuration: grid 1 block 8
// Recursion=0: Hello World from thread 0 block 0
// Recursion=0: Hello World from thread 1 block 0
// Recursion=0: Hello World from thread 2 block 0
// Recursion=0: Hello World from thread 3 block 0
// Recursion=0: Hello World from thread 4 block 0
// Recursion=0: Hello World from thread 5 block 0
// Recursion=0: Hello World from thread 6 block 0
// Recursion=0: Hello World from thread 7 block 0
// -------> nested execution depth: 1
// Recursion=1: Hello World from thread 0 block 0
// Recursion=1: Hello World from thread 1 block 0
// Recursion=1: Hello World from thread 2 block 0
// Recursion=1: Hello World from thread 3 block 0
// -------> nested execution depth: 2
// Recursion=2: Hello World from thread 0 block 0
// Recursion=2: Hello World from thread 1 block 0
// -------> nested execution depth: 3
// Recursion=3: Hello World from thread 0 block 0

// ./nestedHelloWorld 2
// The output of the nested kernel program is now:
// ./nestedHelloWorld 2Execution Configuration: grid 2 block 8
// Recursion=0: Hello World from thread 0 block 1
// Recursion=0: Hello World from thread 1 block 1
// Recursion=0: Hello World from thread 2 block 1
// Recursion=0: Hello World from thread 3 block 1
// Recursion=0: Hello World from thread 4 block 1
// Recursion=0: Hello World from thread 5 block 1
// Recursion=0: Hello World from thread 6 block 1
// Recursion=0: Hello World from thread 7 block 1
// Recursion=0: Hello World from thread 0 block 0
// Recursion=0: Hello World from thread 1 block 0
// Recursion=0: Hello World from thread 2 block 0
// Recursion=0: Hello World from thread 3 block 0
// Recursion=0: Hello World from thread 4 block 0
// Recursion=0: Hello World from thread 5 block 0
// Recursion=0: Hello World from thread 6 block 0
// Recursion=0: Hello World from thread 7 block 0
// -------> nested execution depth: 1
// -------> nested execution depth: 1
// Recursion=1: Hello World from thread 0 block 0
// Recursion=1: Hello World from thread 1 block 0
// Recursion=1: Hello World from thread 2 block 0
// Recursion=1: Hello World from thread 3 block 0
// Recursion=1: Hello World from thread 0 block 0
// Recursion=1: Hello World from thread 1 block 0
// Recursion=1: Hello World from thread 2 block 0
// Recursion=1: Hello World from thread 3 block 0
// -------> nested execution depth: 2
// -------> nested execution depth: 2
// Recursion=2: Hello World from thread 0 block 0
// Recursion=2: Hello World from thread 1 block 0
// Recursion=2: Hello World from thread 0 block 0
// Recursion=2: Hello World from thread 1 block 0
// -------> nested execution depth: 3
// -------> nested execution depth: 3
// Recursion=3: Hello World from thread 0 block 0
// Recursion=3: Hello World from thread 0 block 0