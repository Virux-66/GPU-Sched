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
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>


#include "binomialOptions_common.h"
#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsCPU(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
real randData(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);


    const int OPT_N = MAX_OPTIONS;

    TOptionData optionData[MAX_OPTIONS];
    real
    callValueBS[MAX_OPTIONS],
                callValueGPU[MAX_OPTIONS],
                callValueCPU[MAX_OPTIONS];

    real
    sumDelta, sumRef, gpuTime, errorVal;

    int i;


    srand(123);

    for (i = 0; i < OPT_N; i++)
    {
        optionData[i].S = randData(5.0f, 30.0f);
        optionData[i].X = randData(1.0f, 100.0f);
        optionData[i].T = randData(0.25f, 10.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
        BlackScholesCall(callValueBS[i], optionData[i]);
    }

    cudaDeviceSynchronize();

    binomialOptionsGPU(callValueGPU, optionData, OPT_N);

    cudaDeviceSynchronize();

    printf("[%s] - Shutdown done...\n", argv[0]);



    exit(EXIT_SUCCESS);
}
