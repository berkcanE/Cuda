#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

// nvcc sparseMatrixVectorMultiplication.cu -o sparseMatrixVectorMultiplication -lcusparse


int main() {
    // Initialize cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Define sparse matrix in CSR format
    int m = 4, n = 4, nnz = 9; // Matrix dimensions and number of non-zeros
    float h_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Non-zero values
    int h_columns[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};  // Column indices
    int h_rowIndex[] = {0, 3, 6, 7, 9};             // Row pointers

    // Allocate memory on GPU
    float *d_values, *d_x, *d_y;
    int *d_columns, *d_rowIndex;
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_columns, nnz * sizeof(int));
    cudaMalloc((void**)&d_rowIndex, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowIndex, h_rowIndex, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Define input vector and copy to device
    float h_x[] = {1, 1, 1, 1}; // Example input vector
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    float alpha = 1.0f, beta = 0.0f;

    cusparseCreateCsr(&matA, m, n, nnz, d_rowIndex, d_columns, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // Create dense vector descriptors
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_32F);

    // Allocate workspace buffer
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, 
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Perform SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    // Copy result back to host
    float h_y[m];
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    // Output result
    for (int i = 0; i < m; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_columns);
    cudaFree(d_rowIndex);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    return 0;
}
