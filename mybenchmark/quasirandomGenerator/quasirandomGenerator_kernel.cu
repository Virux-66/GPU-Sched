/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef QUASIRANDOMGENERATOR_KERNEL_CUH
#define QUASIRANDOMGENERATOR_KERNEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include "quasirandomGenerator_common.h"



//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

const int N= 1048576;

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
//__device__ unsigned int* c_Table;//[QRNG_DIMENSIONS][QRNG_RESOLUTION];
__device__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];

static __global__ void quasirandomGeneratorKernel(
    float *d_Output,
    unsigned int seed,
    unsigned int N
)
{
    unsigned int *dimBase = &c_Table[threadIdx.y][0];
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    for (unsigned int pos = tid; pos < N; pos += threadN)
    {
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for (int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if (data & 1)
            {
                result ^= dimBase[bit];
            }

        d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
    }
}

//Table initialization routine

extern "C" void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION])
{
    cudaMemcpyToSymbol(
    /*cudaMemcpy(*/     c_Table,
                        tableCPU,
                        QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int),
                        cudaMemcpyHostToDevice
                    );
}
/*
//Host-side interface
extern "C" void quasirandomGeneratorGPU(float *d_Output, unsigned int seed, unsigned int N)
{
    dim3 threads(128, QRNG_DIMENSIONS);
    quasirandomGeneratorKernel<<<128, threads>>>(d_Output, seed, N);
    getLastCudaError("quasirandomGeneratorKernel() execution failed.\n");
}
*/


////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
__device__ inline float MoroInvCNDgpu(unsigned int x)
{
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;

    float z;

    bool negate = false;

    // Ensure the conversion to floating point will give a value in the
    // range (0,0.5] by restricting the input to the bottom half of the
    // input domain. We will later reflect the result if the input was
    // originally in the top half of the input domain
    if (x >= 0x80000000UL)
    {
        x = 0xffffffffUL - x;
        negate = true;
    }

    // x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
    // Convert to floating point in (0,0.5]
    const float x1 = 1.0f / static_cast<float>(0xffffffffUL);
    const float x2 = x1 / 2.0f;
    float p1 = x * x1 + x2;
    // Convert to floating point in (-0.5,0]
    float p2 = p1 - 0.5f;

    // The input to the Moro inversion is p2 which is in the range
    // (-0.5,0]. This means that our output will be the negative side
    // of the bell curve (which we will reflect if "negate" is true).

    // Main body of the bell curve for |p| < 0.42
    if (p2 > -0.42f)
    {
        z = p2 * p2;
        z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }
    // Special case (Chebychev) for tail
    else
    {
        z = __logf(-__logf(p1));
        z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
    }

    // If the original input (x) was in the top half of the range, reflect
    // to get the positive side of the bell curve
    return negate ? -z : z;
}

////////////////////////////////////////////////////////////////////////////////
// Main kernel. Choose between transforming
// input sequence and uniform ascending (0, 1) sequence
////////////////////////////////////////////////////////////////////////////////
static __global__ void inverseCNDKernel(
    float *d_Output,
    //unsigned int *d_Input,
    unsigned int pathN
)
{
    unsigned int distance = ((unsigned int)-1) / (pathN + 1);
    unsigned int     tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int threadN = MUL(blockDim.x, gridDim.x);

    //Transform input number sequence if it's supplied
/*
    if (d_Input)
    {
        for (unsigned int pos = tid; pos < pathN; pos += threadN)
        {
            unsigned int d = d_Input[pos];
            d_Output[pos] = (float)MoroInvCNDgpu(d);
        }
    }
    //Else generate input uniformly placed samples on the fly
    //and write to destination
    else
    {*/
        for (unsigned int pos = tid; pos < pathN; pos += threadN)
        {
            unsigned int d = (pos + 1) * distance;
            d_Output[pos] = (float)MoroInvCNDgpu(d);
        }
 //   }
}
/*
extern "C" void inverseCNDgpu(float *d_Output, unsigned int *d_Input, unsigned int N)
{
    inverseCNDKernel<<<128, 128>>>(d_Output, d_Input, N);
    getLastCudaError("inverseCNDKernel() execution failed.\n");
}
*/
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]
);

extern "C" float getQuasirandomValue(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION],
    int i,
    int dim
);

extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int p);

extern "C" int cuda_main(int argc, char** argv){
    volatile int64_t num_floatingPoint=10;
    volatile int64_t num_transferredBytes=10;
    volatile float arithmetic_intensity=1.0;
        // Start logs
        printf("%s Starting...\n\n", argv[0]);

        unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];
    
        float *h_OutputGPU;
        float *d_Output;
        //unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];
        //int dim, pos;
        //double delta, ref, sumDelta, sumRef, L1norm, gpuTime;
    
        //StopWatchInterface *hTimer = NULL;
        /* 
        if (sizeof(INT64) != 8)
        {
            printf("sizeof(INT64) != 8\n");
            return 0;
        }
        */
    
        printf("Allocating GPU memory...\n");
        cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float));
        //cudaMalloc((void**)&c_Table,QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int));

        printf("Allocating CPU memory...\n");
        h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));
    
        printf("Initializing QRNG tables...\n\n");
        initQuasirandomGenerator(tableCPU);
    
        initTableGPU(tableCPU);
    
        printf("Testing QRNG...\n\n");
        cudaMemset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(float));
        //int numIterations = 20;
        //int numIterations = 1;
    
        //for (int i = -1; i < numIterations; i++)
        {
            //if (i == 0)
            //{
                cudaDeviceSynchronize();
            //}
    
            //quasirandomGeneratorGPU(d_Output, 0, N);
            

            dim3 threads(128, QRNG_DIMENSIONS);
            quasirandomGeneratorKernel<<<128, threads>>>(d_Output, 0, N);
            getLastCudaError("quasirandomGeneratorKernel() execution failed.\n");

        }
    
        cudaDeviceSynchronize();
    
        printf("\nReading GPU results...\n");
        cudaMemcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(float), cudaMemcpyDeviceToHost);
   /* 
        printf("Comparing to the CPU results...\n\n");
        sumDelta = 0;
        sumRef = 0;
    
        for (dim = 0; dim < QRNG_DIMENSIONS; dim++)
            for (pos = 0; pos < N; pos++)
            {
                ref       = getQuasirandomValue63(pos, dim);
                delta     = (double)h_OutputGPU[dim * N + pos] - ref;
                sumDelta += fabs(delta);
                sumRef   += fabs(ref);
            }
    
        printf("L1 norm: %E\n", sumDelta / sumRef);
  */  

        printf("\nTesting inverseCNDgpu()...\n\n");
        cudaMemset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(float));
    
        //for (int i = -1; i < numIterations; i++)
        {
            //if (i == 0)
            //{
                cudaDeviceSynchronize();
            //}
    
            //inverseCNDgpu(d_Output, NULL, QRNG_DIMENSIONS * N);
            inverseCNDKernel<<<128, 128>>>(d_Output, /*NULL,*/ QRNG_DIMENSIONS * N);
            getLastCudaError("inverseCNDKernel() execution failed.\n");
        }
    
        cudaDeviceSynchronize();
    
        printf("Reading GPU results...\n");
        cudaMemcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(float), cudaMemcpyDeviceToHost);
   /* 
        printf("\nComparing to the CPU results...\n");
        sumDelta = 0;
        sumRef = 0;
        unsigned int distance = ((unsigned int)-1) / (QRNG_DIMENSIONS * N + 1);
    
        for (pos = 0; pos < QRNG_DIMENSIONS * N; pos++)
        {
            unsigned int d = (pos + 1) * distance;
            ref       = MoroInvCNDcpu(d);
            delta     = (double)h_OutputGPU[pos] - ref;
            sumDelta += fabs(delta);
            sumRef   += fabs(ref);
        }
    
        printf("L1 norm: %E\n\n", L1norm = sumDelta / sumRef);
   */ 
        printf("Shutting down...\n");
        free(h_OutputGPU);

        cudaFree(d_Output);

        //cudaFree(c_Table);
    
        //exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);

        exit(EXIT_SUCCESS);
}

#endif
