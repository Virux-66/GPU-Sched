#ifndef _GPU_BEACON_H_
#define _GPU_BEACON_H_

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils.h>

using namespace llvm;

class WrapperPass : public ModulePass {
 private:
  FunctionCallee MallocWrapper;
  FunctionCallee MemcpyWrapper;
  FunctionCallee MemsetWrapper;
  FunctionCallee MemcpyToSymbolWrapper;
  FunctionCallee FreeWrapper;
  FunctionCallee KernelLaunchPrepare;
  FunctionCallee LookUp;
  FunctionCallee debugSgemm;
  FunctionCallee debugLoc;

  void createDebugSgemm(Module &M);
  void createDebugLoc(Module &M);
  void createMallocWrapper(Module &M);
  void createMemcpyWrapper(Module &M);
  void createMemsetWrapper(Module &M);
  void createMemcpyToSymbolWrapper(Module &M);
  void createFreeWrapper(Module &M);
  void createKernelLaunchPrepare(Module &M);
  void createLookUp(Module &M);

  CallInst *getKernelInvokeInst(CallInst *cudaPush);



  void replaceMalloc(CallInst *CI);
  void replaceMemcpy(CallInst *CI);
  void replaceMemset(CallInst *CI);
  void replaceMemcpyToSymbol(CallInst *CI);
  void replaceFree(CallInst *CI);
  void addKernelLaunchPrepare(CallInst *CI);
  void addKernelLaunchPrepareGemm(CallInst *CI);
  void fixKernelParameters(CallInst *CI);
  void fixCublasSgemmParameters(CallInst *CI);


 public:
  static char ID;

  WrapperPass() : ModulePass(ID) {}
  virtual bool doInitialization (Module &) override;
  virtual bool runOnModule(Module &M) override;
};

#endif