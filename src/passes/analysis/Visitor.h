#ifndef __VISITOR_H__
#define __VISITOR_H__

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include "CUDAInfo.h"
#include "Util.h"

using namespace llvm;

class VisitorBase {
 public:
  void visitModule(Module &M);
  virtual void visitFunction(Function &F);
  virtual void visitCallInst(CallInst *CI) {}
  virtual void visitAllocaInst(AllocaInst *SI){}
};

class CUDAVisitor : public VisitorBase {
 private:
  // tracking the kernel invoke information
  std::vector<InvokeInfo> KernelInvokes;

  // tracking the grid initializations
  std::map<Value *, std::vector<GridCtorInfo>> GridCtors;

  // tracking the memory allocation and free operations
  std::map<Value *, std::vector<MemAllocInfo>> MemAllocs;
  std::map<Value *, std::vector<MemFreeInfo>> MemFrees;

  //tracking the arithmetic intensity of a kernel
  llvm::Value* arithmetic_intensity=nullptr;

  // in some case, a copy of grid is used so need to track
  // them where is the initial grid object
  std::map<Value *, Value *> AggMap;
  std::map<Value *, Value *> CoerseMap;

  virtual void visitCallInst(CallInst *CI);
  virtual void visitAllocaInst(AllocaInst *SI);
  bool isDim3Struct(Type *Ty);

 public:
  void collect(Module &M) { visitModule(M); }
  void print();
  Value *getGridDim(InvokeInfo &II);
  Value *getBlockDim(InvokeInfo &II);

  std::vector<InvokeInfo> &getKernelInvokes() { return KernelInvokes; }
  std::vector<MemAllocInfo> &getMemAllocs(Value *Pointer) {
    return MemAllocs[Pointer];
  }

  std::vector<MemFreeInfo> &getMemFrees(Value *Pointer) {
    return MemFrees[Pointer];
  }

  std::vector<GridCtorInfo> &getDimCtor(Value *Pointer) {
    return GridCtors[Pointer];
  }

  llvm::Value *getArithmeticPtr(){
    return arithmetic_intensity;
  }
};

#endif