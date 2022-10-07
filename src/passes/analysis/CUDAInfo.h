#ifndef __CUDA_INFO_H__
#define __CUDA_INFO_H__

#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>

#include <set>

using namespace llvm;

// This class represents a CUDA memory allocation.
// It tracks the target, the size and the call site
// of a CUDA memory alloc operation
class MemAllocInfo {
 private:
  Value *Target;
  Value *Size;
  CallInst *Alloc;

  Value *getTarget();
  Value *getSize();

 public:
  MemAllocInfo() {}
  MemAllocInfo(CallInst *Alloc) : Alloc(Alloc) {
    Target = getTarget();
    Size = getSize();
  }

  Value *getObj() { return Target; }
  CallInst *getCall() { return Alloc; }

  void print(int indent = 0);
};

// This class represents a CUDA memory free operation.
// It tracks the target and the call site of an CUDA
// memory free operation.
class MemFreeInfo {
 private:
  Value *Target;
  CallInst *Call;

  Value *getTarget();

 public:
  MemFreeInfo() {}
  MemFreeInfo(CallInst *Free) : Call(Free) { Target = getTarget(); }
  Value *getObj() { return Target; }
  CallInst *getCall() { return Call; }
  void print() {
    dbgs() << "Target: " << *Target << "\n\tCall: " << *Call << "\n";
  }
};

// This class represents an Dim3 object
class GridCtorInfo {
 private:
  Value *Variable;
  CallInst *Ctor;

  Value *getTarget();

 public:
  GridCtorInfo() {}
  GridCtorInfo(CallInst *Inst) : Ctor(Inst) { Variable = getTarget(); }
  Value *getVar() { return Variable; }
  CallInst *getCall() { return Ctor; }
  void print();
};

class InvokeInfo {
 private:
  Value *gridDim;
  Value *blockDim;
  CallInst *cudaPush;
  CallInst *KernelInvoke;
  std::vector<Value *> MemArgs;

 public:
  InvokeInfo(CallInst *Push = nullptr) : cudaPush(Push) {
    if (cudaPush) {
      retrieveDims();
      retrieveInvoke();
      retrieveMemArgs();
    }
  }

  void print();

  std::vector<Value *> &getMemOperands() { return MemArgs; }
  Value *getGridDimObjs() { return gridDim; }
  Value *getBlockDimObj() { return blockDim; }
  CallInst *getPush() { return cudaPush; }

 private:
  Value *getBase(Value *);
  // rtrieveDims is to get the dim argumetns of an kernel execution
  void retrieveDims();

  // retrieveInvok is to get the CallInst of kernel wrapper
  void retrieveInvoke();

  // retrieveMemArgs is to find the memory objects (allocated by
  // cudaMemoryAlloc) used by the kernel
  void retrieveMemArgs();
};

static int globalID = 0;
class CUDATask {
 private:
  int TaskTypeID;  // 1 --> unit task, 2 --> complex task
  int TaskID;

 public:
  CUDATask(int TyID) {
    TaskTypeID = TyID;
    TaskID = -1;
  };

  int getTaskID() {
    if (TaskID == -1) TaskID = globalID++;
    return TaskID;
  }

  bool isComplexTask() { return TaskTypeID == 2; }
  bool isUnitTask() { return TaskTypeID == 1; }
  virtual std::set<CallInst *> getMemAllocOps() = 0;
  virtual std::set<CallInst *> getMemFreeOps() = 0;
  virtual std::vector<Value *> getCUDAMemSize() = 0;
  virtual std::vector<Value *> getGridDims() = 0;
  virtual std::vector<Value *> getBlockDims() = 0;

  virtual void print() = 0;
};

// a CUDAUnitTask contains exactly one cuda kernel call as well as related
// cuda memory
class CUDAUnitTask : public CUDATask {
 private:
  CallInst *gridCtor;   // the info where gridDim is initialized
  CallInst *blockCtor;  // the info where blockDim is initialized
  CallInst *KernelInvok;
  std::set<CallInst *> MemAllocOps;
  std::set<CallInst *> MemFreeOps;

 public:
  CUDAUnitTask(const CUDAUnitTask &UT)
      : CUDATask(1), MemAllocOps(UT.MemAllocOps), MemFreeOps(UT.MemFreeOps) {
    gridCtor = UT.gridCtor;
    blockCtor = UT.blockCtor;
    KernelInvok = UT.KernelInvok;
  }

  CUDAUnitTask(CallInst *G, CallInst *B, CallInst *K, std::set<CallInst *> &M,
               std::set<CallInst *> &F)
      : CUDATask(1), MemAllocOps(M), MemFreeOps(F) {
    gridCtor = G;
    blockCtor = B;
    KernelInvok = K;
  }

  CallInst *getGridCtor() { return gridCtor; }
  CallInst *getBlockCtor() { return blockCtor; }
  std::vector<Value *> getGridDims();
  std::vector<Value *> getBlockDims();

  std::set<CallInst *> getMemAllocOps() { return MemAllocOps; }
  std::set<CallInst *> getMemFreeOps() { return MemFreeOps; }
  std::vector<Value *> getCUDAMemSize();
  void print();
};

// A CDUAComplexTask contains several CUDA operations. In high-level, these
// operations are for kernel executions or for preparation of kernel
// executions, including e.g., cudaMalloc, __cudaPushConfiguration,
// and cudaFree and so on.
class CUDAComplexTask : public CUDATask {
 private:
  std::vector<CUDAUnitTask> SubTasks;

 public:
  CUDAComplexTask(std::vector<CUDAUnitTask> &UnitTasks)
      : CUDATask(2), SubTasks(UnitTasks) {}
  std::set<CallInst *> getMemAllocOps();
  std::set<CallInst *> getMemFreeOps();
  std::vector<Value *> getCUDAMemSize();
  std::vector<Value *> getGridDims();
  std::vector<Value *> getBlockDims();
  void print();
};

#endif