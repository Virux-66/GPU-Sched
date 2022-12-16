/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
    volatile int64_t num_floatingPoint=100000000;
    volatile int64_t num_transferredBytes=1288490188;
    volatile float arithmetic_intensity=0.0776;

    printf("[%s] - Starting...\n",argv[0]);

    // Error code to check return values for CUDA calls
    //cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    //int numElements = 50000;
    int numElements = 100000000;
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate the device input vector A
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    // Allocate the device input vector B
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);


    // Allocate the device output vector C
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);


    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    // Launch the Vector Add CUDA Kernel

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaGetLastError();


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    // Free device global memory
    cudaFree(d_A);



    cudaFree(d_B);



    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    printf("[%s] - Shutdown done...\n",argv[0]);
    return 0;
}

