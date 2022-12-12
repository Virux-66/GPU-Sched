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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


//#include <helper_functions.h>   // helper functions for string parsing
//#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"
#include <stdio.h>
////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 80000000;  //If we want scale up this program, we just need to increase OPT_N
const int  NUM_ITERATIONS = 512;


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    volatile int64_t num_floatingPoint=5751322851;
    volatile int64_t num_transferredBytes=1677721.6;
    volatile float arithmetic_intensity=3.3477;
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    int i;

    //findCudaDevice(argc, (const char **)argv);


    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    int grid=((OPT_N/2)+128-1)/128;
    int block=128;

    printf("...allocating GPU memory for options.\n");
    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice);
    printf("Data init done.\n\n");


    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    cudaDeviceSynchronize();

    BlackScholesGPU<<<grid, 128/*480, 128*/>>>(
        (float2 *)d_CallResult,
        (float2 *)d_PutResult,
        (float2 *)d_StockPrice,
        (float2 *)d_OptionStrike,
        (float2 *)d_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

    cudaDeviceSynchronize();


    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost);


    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    printf("Shutdown done.\n");

    exit(EXIT_SUCCESS);
}
