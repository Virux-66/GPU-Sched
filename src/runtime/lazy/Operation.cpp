
#include "Operation.h"

#include <iostream>

cudaError_t MallocOp::perform() {
  uint64_t* fake_addr = (uint64_t*)devMem->ptr;
  cudaError_t err = cudaMalloc(&(devMem->ptr), devMem->size);
#if DEBUG
  fprintf(stderr, "    replayed cudaMalloc(fake: %p, valid: %p)\n", fake_addr,
          devMem->ptr);
#endif
  return err;
}

cudaError_t MemcpyOp::perform() {
  cudaError_t err = cudaMemcpy(devMem->ptr, buf, size, kind);
  free(buf);
#if DEBUG
  fprintf(stderr,
          "    replayed cudaMemcpy(dst: %p, src: %p, size: %ld, kind: %d)\n",
          devMem->ptr, src, size, kind);
#endif
  return err;
}

cudaError_t MemcpyToSymbolOp::perform() {
#if DEBUG
  fprintf(stderr,
          "    relay cudaMemcpyToSymbol (sym: %p, src: %p, size: %ld, offset: "
          "%ld, kind: %d)\n",
          symbol, buf, count, offset, kind);
#endif
  cudaError_t err = cudaMemcpyToSymbol(symbol, buf, count, offset, kind);
  free(buf);
  return err;
}

cudaError_t MemsetOp::perform() {
#if DEBUG
  fprintf(stderr, "    relay cudaMemset (ptr: %p, value: %d, size: %ld)\n",
          devMem->ptr, value, count);
#endif
  return cudaMemset(devMem->ptr, value, count);
}
