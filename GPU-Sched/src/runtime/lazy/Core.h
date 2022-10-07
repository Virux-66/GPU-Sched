#ifndef _CORE_H_
#define _CORE_H_

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>

#include "../bemps/bemps.hpp"
#include "Operation.h"

using namespace std;

class Runtime {
 private:
  bool issue;  // need to issue an beacon ?
  map<uint64_t, MObject*> MemObjects; // this tracks the memobjects pending for creation
  map<uint64_t, std::vector<Operation*>> CudaMemOps; // the tracks the cuda memory operations for each memobjects

  unordered_set<uint64_t> ActiveObjects;
  map<uint64_t, uint64_t> AllocatedMap; // this tracks the pair of fake_addr to real_addr
  map<uint64_t, uint64_t> SizeMap; // this tracks the size of each memory object, indexed with fake_addr
  map<uint64_t, uint64_t> ReverseAllocatedMap; // this tracks the pair of real_addr to fake_addr
  std::vector<Operation*> DeviceDependentOps;

 public:
  Runtime() : issue(true) {}
  bool toIssue() { return issue; }
  void enableIssue() { issue = true; }
  void disableIssue() { issue = false; }

  bool isWithinAllocatedRegion(void *ptr);
  bool isAllocated(void* ptr);
  void* getValidAddrforFakeAddr(void* ptr);
  void* getValidAddr(void *addr);

  cudaError_t registerMallocOp(void** holder, size_t size);
  cudaError_t registerMemcpyOp(void* dst, void* src, size_t size,
                        enum cudaMemcpyKind k);
  cudaError_t registerMemcpyToSymbleOp(char* symble, void* src, size_t s, size_t o,
                                enum cudaMemcpyKind kind);
  cudaError_t registerMemsetOp(void *ptr, int val, size_t size);

  int64_t getAggMemSize();
  cudaError_t prepare();
  cudaError_t free(void* devPtr);
};

#endif