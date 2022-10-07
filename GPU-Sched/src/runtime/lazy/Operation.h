#ifndef __OPERATION_H_
#define __OPERATION_H_

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

typedef enum oper {
  CUDA_MALLOC,
  CUDA_MEMCPY,
  CUDA_MEMCPY_TO_SYMBOL,
  CUDA_MEMSET
} opr_t;

// MOject is to represent a memory region
struct MObject {
  void *ptr;
  size_t size;
  MObject(void *ptr, size_t s) : ptr(ptr), size(s) {}
};

class Operation {
 private:
  opr_t op;

 protected:
  MObject *devMem;

 public:
  Operation(opr_t op, MObject *obj) : op(op), devMem(obj) {}
  bool isMalloc() { return op == CUDA_MALLOC; }
  bool isMemcpy() { return op == CUDA_MEMCPY; }
  bool isMemset() { return op == CUDA_MEMSET; }
  bool isMemcpyToSymbol() { return op == CUDA_MEMCPY_TO_SYMBOL; }
  virtual cudaError_t perform() = 0;
};

class MallocOp : public Operation {
 public:
  MallocOp(MObject *obj) : Operation(CUDA_MALLOC, obj) {}
  cudaError_t perform() override;
};

// here we only interested in host to device copy
class MemcpyOp : public Operation {
 private:
  void *buf;
  void *src;
  size_t size;
  enum cudaMemcpyKind kind;

 public:
  MemcpyOp(void *src, MObject *dst, size_t s, enum cudaMemcpyKind k)
      : src(src),size(s), kind(k), Operation(CUDA_MEMCPY, dst) {
        buf = malloc(size);
        memcpy(buf, src, size);
      }
  cudaError_t perform() override;
};

class MemcpyToSymbolOp : public Operation {
 private:
  char *symbol;
  void *buf;
  size_t count;
  size_t offset;
  enum cudaMemcpyKind kind;

 public:
  MemcpyToSymbolOp(char *sym, void *src, size_t s, size_t offset = 0,
                   enum cudaMemcpyKind k = cudaMemcpyHostToDevice)
      : symbol(sym),
        count(s),
        offset(offset),
        kind(k),
        Operation(CUDA_MEMCPY_TO_SYMBOL, nullptr) {
    buf = malloc(count);
    std::memcpy(buf, src, count);
  }
  cudaError_t perform() override;
};

class MemsetOp : public Operation {
 private:
  int value;
  size_t count;

 public:
  MemsetOp(MObject *obj, int val, size_t s)
      : value(val), count(s), Operation(CUDA_MEMSET, obj) {}
  cudaError_t perform() override;
};

#endif