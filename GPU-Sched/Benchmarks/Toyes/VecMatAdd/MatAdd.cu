/**
 * This is a toy program from CUDA tutorial, it is to compute the addition on
 * two square matrixes, Here I use it to study some basic built-in data
 * structures provided by CUDA
 */

#include <math.h>
#include <stdio.h>

#define N 512
#define BLK 32

__global__ void MatAdd(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) C[i * N + j] = A[i * N + j] + B[i * N + j];
}

int main(int argc, char **argv) {
  // A and B are input data, C is to store output data
  float A[N][N], B[N][N], C[N][N];

  size_t size = N * N * sizeof(float);

  printf("size: %ld\n", size);

  // initialize the input data
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = sin(i + j) * sin(i + j);
      B[i][j] = cos(i + j) * cos(i + j);
      // C[i][j] = A[i][j] + B[i][j];
    }
  }

  // correspoinding memory for data in device
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlock(BLK, BLK);
  dim3 BlocksPerGrid(N / BLK, N / BLK);
  MatAdd<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < BLK; i++) {
    for (int j = 0; j < BLK; j++) {
      printf("%.2f ", C[i][j]);
    }
    printf("\n");
  }

  return 0;
}