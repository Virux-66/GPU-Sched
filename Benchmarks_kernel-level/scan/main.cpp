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


//#include <cuda_runtime.h>

//#include "scan_common.h"

extern "C" int cuda_main(int argc,char** argv);

int main(int argc, char **argv)
{
    return cuda_main(argc,argv);
}
