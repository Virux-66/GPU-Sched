/**
 * This is a toy program from CUDA tutorial, it is to compute the addition on
 * two square matrixes, Here I use it to study some basic built-in data
 * structures provided by CUDA
 */

#include <math.h>
#include <stdio.h>

#define N 4096
#define BLK 32

__global__ void VecAdd(double *A, double *B, double *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
  printf("A[%d] (%f) + B[%d] (%f) = C[%d] (%f)\n", i, A[i], i, B[i], i, C[i]);
}

void VecAddWrapper(double *d_A, double *d_B, double *d_C) {
  dim3 ThreadsPerBlock(BLK);
  dim3 BlocksPerGrid(N/BLK);
  VecAdd<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C);
}

int main(int argc, char **argv) {
  // A and B are input data, C is to store output data
  double A[N], B[N], C[N], sum = 0;
  size_t size = N * sizeof(double);

  // initialize the input data
  for (int i = 0; i < N; i++) {
    A[i] = sin(i) * sin(i);
    B[i] = cos(i) * cos(i);
  }

  // correspoinding memory for data in device
  double *d_src[2], *d_res;
  cudaMalloc(&d_src[0], size);
  cudaMalloc(&d_src[1], size);
  cudaMalloc(&d_res, size);

  printf("Init: %p, %p, %p\n", d_src[0], d_src[1], d_res);

  cudaMemcpy(d_src[0], A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_src[1], B, size, cudaMemcpyHostToDevice);

  printf("Maturized: %p, %p, %p\n", d_src[0], d_src[1], d_res);

  // cudaMemcpy(d_src[1], d_res, size, cudaMemcpyDeviceToDevice);
  // VecAdd<<<BlocksPerGrid, ThreadsPerBlock>>>(d_src[0], d_src[1], d_res);
  VecAddWrapper(d_src[0], d_src[1], d_res);

  cudaMemcpy(C, d_res, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    if (i % 64 == 0) printf("C[%d] = %f\n", i, C[i]);
    sum+= C[i];
  }
  printf("sum = %f\n", sum);


  cudaFree(d_src[0]);
  cudaFree(d_src[1]);
  cudaFree(d_res);
  return 0;
}
