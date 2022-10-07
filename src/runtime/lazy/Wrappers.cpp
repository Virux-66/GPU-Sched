
#ifndef _LAZY_CUDA_OP_H_
#define _LAZY_CUDA_OP_H_

#include "Core.h"


static Runtime R;
static int id = 0;



extern bool is_fake_addr(void* ptr);

extern "C" void debugSgemm(void *A, void *a, void *B, void *b, void *C, void *c) {
  fprintf(stdout, "debugSgemm --- A: %p, B: %p, C: %p (a: %p, b: %p, c: %p)\n\n", A, B, C, a, b, c);
}

extern "C" void debugLoc(char *function, char *kernel) {
  fprintf(stdout, "debugLoc --- Function: %s, Kernel: %s\n\n", function, kernel);
}

extern "C" cudaError_t cudaMallocWrapper(void** devPtr, size_t size) {
#if NOLAZY
  cudaError_t err = cudaMalloc(devPtr, size);
  fprintf(stderr, "cudaMalloc: %p (%ld)\n", *devPtr, size);
  return err;
#else

  R.registerMallocOp(devPtr, size);
#if DEBUG
  fprintf(stderr, "\nDelay a cudaMalloc(holder: %p, fake: %p)\n", devPtr,
          *devPtr);
#endif
  return cudaSuccess;

#endif
}

extern "C" cudaError_t cudaMemcpyWrapper(void* dst, void* src, size_t count,
                                         enum cudaMemcpyKind kind) {
#if NOLAZY
  fprintf(stderr, "cudaMemcpy (%p, %p, %ld, %d)\n", dst, src, count, kind);
  return cudaMemcpy(dst, src, count, kind);
#endif
  if (is_fake_addr(dst) && R.isAllocated(dst))
    dst = R.getValidAddrforFakeAddr(dst);

  if (is_fake_addr(src) && R.isAllocated(src))
    src = R.getValidAddrforFakeAddr(src);

  if ((kind == cudaMemcpyHostToDevice && !is_fake_addr(dst)) ||
      (kind == cudaMemcpyDeviceToDevice && !is_fake_addr(dst) &&
       !is_fake_addr(src)) ||
      kind == cudaMemcpyDeviceToHost) {
#if DEBUG
    fprintf(stderr,
            "\nPerform a cudaMemcpy(dst: %p, src: %p, size: %ld, kind: %d)\n",
            dst, src, count, kind);
#endif
    return cudaMemcpy(dst, src, count, kind);
  } else {
    R.registerMemcpyOp(dst, src, count, kind);
#if DEBUG
    fprintf(stderr,
            "\nDelay a cudaMemcpy(dst: %p, src: %p, size: %ld, kind: %d)\n",
            dst, src, count, kind);
#endif
    return cudaSuccess;
  }
}

extern "C" cudaError_t cudaMemsetWrapper(void* ptr, int val, size_t size) {
#if NOLAZY
  fprintf(stderr, "cudaMemset(%p, %d, %ld)\n", ptr, val, size);
  return cudaMemset(ptr, val, size);
#endif
  if (is_fake_addr(ptr) && R.isAllocated(ptr))
    ptr = R.getValidAddrforFakeAddr(ptr);
  if (!is_fake_addr(ptr))
    return cudaMemset(ptr, val, size);
  else {
#if DEBUG
    fprintf(stderr, "\nDelay a cudaMemset(%p, %d, %ld)\n", ptr, val, size);
#endif
    return R.registerMemsetOp(ptr, val, size);
  }
}

extern "C" cudaError_t cudaMemcpyToSymbolWrapper(char* sym, void* src,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind) {
#if NOLAZY
  return cudaMemcpyToSymbol(sym, src, count, offset, kind);
#endif
#if DEBUG
  fprintf(stderr, "\nDelay a cudaMemcpyToSymbol(%p, %p, %ld, %ld, %d)\n", sym,
          src, count, offset, kind);
#endif
  R.registerMemcpyToSymbleOp(sym, src, count, offset, kind);
  return cudaSuccess;
}

extern "C" cudaError_t cudaKernelLaunchPrepare(uint64_t gxy, int gz,
                                               uint64_t bxy, int bz) {
#if NOLAZY
  return cudaSuccess;
#endif

  // TODO: here add code to call beacon
#define U32X(v) (int)((v & 0xFFFFFFFF00000000LL) >> 32)
#define U32Y(v) (int)(v & 0xFFFFFFFFLL)
  int gx = U32X(gxy);
  int gy = U32Y(gxy);
  int bx = U32X(bxy);
  int by = U32Y(bxy);
  int64_t membytes = R.getAggMemSize();

#if DEBUG
  printf(
      "Prepare for a new kernel launch: \n\tgridDim(gx: %d, gy: %d, gz: %d)"
      "\n\tblockDim(bx: %d, by: %d, bz: %d) \n\tmem: %ld, toIssue: %d"
      "\nPreparing ...\n",
      gx, gy, gz, bx, by, bz, membytes, R.toIssue());
#endif

  if (R.toIssue()) {
    bemps_begin(id, gx, gy, gz, bx, by, bz, membytes);
    R.disableIssue();
  }
  cudaError_t err = R.prepare();

  if (err != cudaSuccess) {
#if DEBUG
    printf("\nPrepare Failed!!!\n\n\n");
#endif
    exit(EXIT_FAILURE);
  }
#if DEBUG
  printf("\nPrepare Succeed!!!\n\n\n");
#endif
  return err;
}

extern "C" cudaError_t cudaFreeWrapper(void* devPtr) {
#if NOLAZY
  return cudaFree(devPtr);
#endif
#if DEBUG
  printf("\ncudaFree(%p)\n", devPtr);
#endif
  cudaError_t err = R.free(devPtr);
  if (R.toIssue()) {
    bemps_free(id);
    id++;
  }
  return err;
}

extern "C" void * lookup(void * addr) {
  void * res = R.getValidAddr(addr);
#if DEBUG
  printf("\nLookup: %p --> %p\n", addr, res);
#endif
  return res;
}

#endif