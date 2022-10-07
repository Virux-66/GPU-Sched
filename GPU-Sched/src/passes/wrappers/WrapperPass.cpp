
#include "WrapperPass.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

using namespace llvm;

void WrapperPass::createDebugSgemm(Module &M) {
  auto &ctx = M.getContext();
  Type *paramty = Type::getInt8PtrTy(ctx);
  Type *retTy = Type::getVoidTy(ctx);
  FunctionType *FTy = FunctionType::get(retTy, {paramty, paramty, paramty, paramty, paramty, paramty}, false);
  debugSgemm = M.getOrInsertFunction("debugSgemm", FTy);
}

void WrapperPass::createDebugLoc(Module &M) {
  auto &ctx = M.getContext();
  Type *paramty = Type::getInt8PtrTy(ctx);
  Type *retTy = Type::getVoidTy(ctx);
  FunctionType *FTy = FunctionType::get(retTy, {paramty, paramty}, false);
  debugLoc = M.getOrInsertFunction("debugLoc", FTy);
}

void WrapperPass::createMallocWrapper(Module &M)
{
  auto &ctx = M.getContext();
  Type *retTy = Type::getInt32Ty(ctx);
  Type *ptrTy = Type::getInt8PtrTy(ctx)->getPointerTo();
  Type *sizeTy = Type::getInt64Ty(ctx);
  FunctionType *FTy = FunctionType::get(retTy, {ptrTy, sizeTy}, false);
  MallocWrapper = M.getOrInsertFunction("cudaMallocWrapper", FTy);
}

void WrapperPass::createMemcpyWrapper(Module &M)
{
  auto &ctx = M.getContext();
  Type *retTy = Type::getInt32Ty(ctx);
  Type *dstTy = Type::getInt8PtrTy(ctx);
  Type *srcTy = Type::getInt8PtrTy(ctx);
  Type *sizeTy = Type::getInt64Ty(ctx);
  Type *kindTy = Type::getInt32Ty(ctx);
  FunctionType *FTy =
      FunctionType::get(retTy, {dstTy, srcTy, sizeTy, kindTy}, false);
  MemcpyWrapper = M.getOrInsertFunction("cudaMemcpyWrapper", FTy);
}

void WrapperPass::createMemsetWrapper(Module &M)
{
  auto &ctx = M.getContext();
  Type *retTy = Type::getInt32Ty(ctx);
  Type *dstTy = Type::getInt8PtrTy(ctx);
  Type *valTy = Type::getInt32Ty(ctx);
  Type *sizeTy = Type::getInt64Ty(ctx);
  FunctionType *FTy = FunctionType::get(retTy, {dstTy, valTy, sizeTy}, false);
  MemsetWrapper = M.getOrInsertFunction("cudaMemsetWrapper", FTy);
}

void WrapperPass::createMemcpyToSymbolWrapper(Module &M)
{
  auto &ctx = M.getContext();
  Type *retTy = Type::getInt32Ty(ctx);
  Type *dstTy = Type::getInt8PtrTy(ctx);
  Type *srcTy = Type::getInt8PtrTy(ctx);
  Type *sizeTy = Type::getInt64Ty(ctx);
  Type *kindTy = Type::getInt32Ty(ctx);
  FunctionType *FTy =
      FunctionType::get(retTy, {dstTy, srcTy, sizeTy, sizeTy, kindTy}, false);
  MemcpyToSymbolWrapper =
      M.getOrInsertFunction("cudaMemcpyToSymbolWrapper", FTy);
}

void WrapperPass::createFreeWrapper(Module &M)
{
  auto &ctx = M.getContext();
  Type *retTy = Type::getInt32Ty(ctx);
  Type *ptrTy = Type::getInt8PtrTy(ctx);
  FunctionType *FTy = FunctionType::get(retTy, {ptrTy}, false);
  FreeWrapper = M.getOrInsertFunction("cudaFreeWrapper", FTy);
}

void WrapperPass::createKernelLaunchPrepare(Module &M)
{
  auto &ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(ctx);
  Type *Int64Ty = Type::getInt64Ty(ctx);
  FunctionType *FTy =
      FunctionType::get(Int32Ty, {Int64Ty, Int32Ty, Int64Ty, Int32Ty}, false);
  KernelLaunchPrepare = M.getOrInsertFunction("cudaKernelLaunchPrepare", FTy);
}

void WrapperPass::createLookUp(Module &M)
{
  auto &ctx = M.getContext();
  Type *Int8PtrTy = Type::getInt8PtrTy(ctx);
  FunctionType *FTy = FunctionType::get(Int8PtrTy, {Int8PtrTy}, false);
  LookUp = M.getOrInsertFunction("lookup", FTy);
}

void WrapperPass::replaceMalloc(CallInst *CI)
{
  dbgs() << "repalcing Malloc : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 2> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  auto ret = IRB.CreateCall(MallocWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceMemcpy(CallInst *CI)
{
  dbgs() << "repalcing Memcpy : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\t" << *CI->getArgOperand(2)
         << "\n\t" << *CI->getArgOperand(3) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  args.push_back(CI->getArgOperand(3));
  auto ret = IRB.CreateCall(MemcpyWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceMemset(CallInst *CI)
{
  dbgs() << "repalcing Memset : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\t" << *CI->getArgOperand(2)
         << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  auto ret = IRB.CreateCall(MemsetWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceMemcpyToSymbol(CallInst *CI)
{
  dbgs() << "repalcing MemcpyToSymbol : " << *CI << "\n\t"
         << *CI->getArgOperand(0) << "\n\t" << *CI->getArgOperand(1) << "\n\t"
         << *CI->getArgOperand(2) << "\n\t" << *CI->getArgOperand(3) << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  args.push_back(CI->getArgOperand(3));
  args.push_back(CI->getArgOperand(4));
  auto ret = IRB.CreateCall(MemcpyToSymbolWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::replaceFree(CallInst *CI)
{
  dbgs() << "repalcing Free : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\n";
  IRBuilder<NoFolder> IRB(CI->getContext());
  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 2> args;
  args.push_back(CI->getArgOperand(0));
  auto ret = IRB.CreateCall(FreeWrapper, args);
  CI->replaceAllUsesWith(ret);
}

void WrapperPass::addKernelLaunchPrepare(CallInst *CI)
{
  dbgs() << "adding LaunchPrepare : " << *CI << "\n\t" << *CI->getArgOperand(0)
         << "\n\t" << *CI->getArgOperand(1) << "\n\t" << *CI->getArgOperand(2)
         << "\n\t" << *CI->getArgOperand(3) << "\n\n";

  CallInst *Invoke = getKernelInvokeInst(CI);

  auto &ctx = CI->getContext();
  IRBuilder<NoFolder> IRB(ctx);
  IRB.SetInsertPoint(CI);

  // Type *Int64Ty = Type::getInt64Ty(ctx);
  // Constant *FuncName = ConstantDataArray::getString(ctx, CI->getFunction()->getName(), true);
  // Constant *KernelName = ConstantDataArray::getString(ctx, Invoke->getName(), true);

  // Module *M = CI->getModule();

  // GlobalVariable *F = new GlobalVariable(*M, FuncName->getType(), true, GlobalValue::PrivateLinkage, FuncName);
  // GlobalVariable *K = new GlobalVariable(*M, KernelName->getType(), true, GlobalValue::PrivateLinkage, KernelName);

  // Constant *idx = ConstantInt::get(Int64Ty, 0, true);
  // Value * Fb = IRB.CreateGEP(F, {idx, idx});
  // Value * Kb = IRB.CreateGEP(K, {idx, idx});

  // IRB.CreateCall(debugLoc, {Fb, Kb});

  SmallVector<Value *, 4> args;
  args.push_back(CI->getArgOperand(0));
  args.push_back(CI->getArgOperand(1));
  args.push_back(CI->getArgOperand(2));
  args.push_back(CI->getArgOperand(3));
  IRB.CreateCall(KernelLaunchPrepare, args);
}

void WrapperPass::addKernelLaunchPrepareGemm(CallInst *CI)
{
  dbgs() << "adding LaunchPrepare for GEMM: " << *CI << "\n";
  auto &ctx = CI->getContext();
  Type *Int32Ty = Type::getInt32Ty(ctx);
  Type *Int64Ty = Type::getInt64Ty(ctx);
  IRBuilder<NoFolder> IRB(ctx);

  IRB.SetInsertPoint(CI);
  SmallVector<Value *, 4> args;

  args.push_back(ConstantInt::get(Int64Ty, 0, true));
  args.push_back(ConstantInt::get(Int32Ty, 0, true));
  args.push_back(ConstantInt::get(Int64Ty, 0, true));
  args.push_back(ConstantInt::get(Int32Ty, 0, true));

  IRB.CreateCall(KernelLaunchPrepare, args);
}

void WrapperPass::fixKernelParameters(CallInst *cudaPush)
{
  CallInst *Invoke = getKernelInvokeInst(cudaPush);
  assert(Invoke);

  auto &ctx = cudaPush->getContext();
  IRBuilder<NoFolder> IRB(ctx);
  IRB.SetInsertPoint(Invoke);

  dbgs() << "Kernel Invoke: " << *Invoke;

  for (int i = 0; i < Invoke->getNumArgOperands(); i++)
  {
    auto arg = Invoke->getArgOperand(i);
    if (arg->getType()->isPointerTy())
    {
      SmallVector<Value *, 1> ops;
      auto tmp = IRB.CreateBitCast(arg, Type::getInt8PtrTy(ctx));
      ops.push_back(tmp);
      Value *lookup = IRB.CreateCall(LookUp, ops);
      dbgs() << "\n\tfix [" << i << "]: " << *arg << " ---> " << *tmp << "\n";

      if (lookup->getType() != arg->getType())
        lookup = IRB.CreateBitCast(lookup, arg->getType());
      Invoke->replaceUsesOfWith(arg, lookup);
    }
  }
  Invoke->getParent()->dump();
}

void WrapperPass::fixCublasSgemmParameters(CallInst *cublasSgemm)
{
  auto &ctx = cublasSgemm->getContext();
  IRBuilder<NoFolder> IRB(ctx);
  IRB.SetInsertPoint(cublasSgemm);

  SmallVector<int, 3> params = {7, 9, 12};
  // the 7th, 9th, and 12nd params of cublaSgemm refers to GPU memory

  // SmallVector<Value *, 6> DbgSgemmParams;

  for (auto i : params)
  {
    SmallVector<Value *, 1> ops;
    auto arg = cublasSgemm->getArgOperand(i);
    auto tmp = IRB.CreateBitCast(arg, Type::getInt8PtrTy(ctx));
    ops.push_back(tmp);
    Value *lookup = IRB.CreateCall(LookUp, ops);
    // DbgSgemmParams.push_back(tmp);
    // DbgSgemmParams.push_back(lookup);
    dbgs() << "\n\tfix sgemm[" << i << "]: " << *arg << " ---> " << *tmp << "\n";
    if (lookup->getType() != arg->getType())
      lookup = IRB.CreateBitCast(lookup, arg->getType());
    cublasSgemm->replaceUsesOfWith(arg, lookup);
  }
  // IRB.CreateCall(debugSgemm, DbgSgemmParams);
}

CallInst *WrapperPass::getKernelInvokeInst(CallInst *cudaPush)
{
  int idx = 0;
  Instruction *tmp = cudaPush->getNextNonDebugInstruction();

  while (!isa<CallInst>(tmp))
  {
    if (isa<BranchInst>(tmp))
    {
      tmp = dyn_cast<BranchInst>(tmp)->getSuccessor(idx)->getFirstNonPHIOrDbg();
    }
    else if (auto Cmp = dyn_cast<CmpInst>(tmp))
    {
      idx = 1 - Cmp->isTrueWhenEqual();
      tmp = tmp->getNextNonDebugInstruction();
    }
    else
    {
      tmp = tmp->getNextNonDebugInstruction();
    }
  }
  return dyn_cast<CallInst>(tmp);
}

bool WrapperPass::doInitialization(Module &M)
{
  auto &ctx = M.getContext();

  createDebugSgemm(M);
  createDebugLoc(M);

  // declare cudaMallocWrapper
  createMallocWrapper(M);

  // declare cudaMemcpyWrapper
  createMemcpyWrapper(M);
  createMemsetWrapper(M);
  createMemcpyToSymbolWrapper(M);
  // declare cudaFreeWrapper()
  createFreeWrapper(M);
  // declare cudaKernelLaunchPrepare()
  createKernelLaunchPrepare(M);
  createLookUp(M);

  return true;
}

bool WrapperPass::runOnModule(Module &M)
{
  Module::FunctionListType &Funcs = M.getFunctionList();
  for (auto ft = Funcs.begin(); ft != Funcs.end(); ft++)
  {;
    Function &F = *ft;
    if (F.isIntrinsic() || F.isDeclaration())
      continue;

    SmallVector<CallInst *, 4> ToBeRemoved;
    for (auto it = inst_begin(F); it != inst_end(F); it++)
    {
      Instruction *I = &*it;
      auto CI = dyn_cast<CallInst>(I);

      if (!CI)
        continue;

      auto Callee = CI->getCalledFunction();
      if (!Callee)
        continue;

      auto name = Callee->getName();

      if (name == "cudaMalloc")
      {
        replaceMalloc(CI);
        ToBeRemoved.push_back(CI);
      }
      else if (name == "cudaMemcpy")
      {
        replaceMemcpy(CI);
        ToBeRemoved.push_back(CI);
      }
      else if (name == "cudaMemcpyToSymbol")
      {
        replaceMemcpyToSymbol(CI);
        ToBeRemoved.push_back(CI);
      }
      else if (name == "cudaMemset")
      {
        replaceMemset(CI);
        ToBeRemoved.push_back(CI);
      }
      else if (name == "cudaFree")
      {
        replaceFree(CI);
        ToBeRemoved.push_back(CI);
      }
      else if (name == "__cudaPushCallConfiguration")
      {
        addKernelLaunchPrepare(CI);
        fixKernelParameters(CI);
      }
      else if (name == "cublasSgemm_v2")
      {
        addKernelLaunchPrepareGemm(CI);
        fixCublasSgemmParameters(CI);
      }
      else if (name.startswith("cublas") && name != "cublasCreate_v2")
      {
        dbgs() << "Unhandled CUBLAS Call: " << *CI;
        llvm_unreachable("Unreachable point");
      }
    }
    for (auto CI : ToBeRemoved)
      CI->eraseFromParent();
  }
  return true;
}

char WrapperPass::ID = 0;

#if 1
static RegisterPass<WrapperPass> X("WP", "WrapperPass", false, false);

#else

static void registerWP(const PassManagerBuilder &,
                       legacy::PassManagerBase &PM)
{
  PM.add(new WrapperPass());
}

// Use EP_OptimizerLast to make sure the pass is run after all other
// optimization passes, such that the debug data is not removed by others
static RegisterStandardPasses RegisterMyPass(
    PassManagerBuilder::EP_OptimizerLast, registerWP);
#endif