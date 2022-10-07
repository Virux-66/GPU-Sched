
#include "GPUBeacon.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "CUDAInfo.h"

using namespace llvm;

int GPUBeaconPass::getDistance(Instruction *S, Instruction *E) {
  int dist = 0;
  auto Func = S->getFunction();
  bool counter = false;

  for (inst_iterator it = inst_begin(*Func); it != inst_end(*Func); it++) {
    Instruction *I = &*it;
    if (I == S) counter = true;
    if (counter) dist++;
    if (I == E) return dist;
  }
  return INT_MAX;
}

bool GPUBeaconPass::runOnModule(Module &M) {
  SmallVector<Type *, 4> ParamTys;
  auto &Ctx = M.getContext();

  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for TaskID
  FunctionType *ETy = FunctionType::get(Type::getVoidTy(Ctx), ParamTys, false);
  BeaconRelease = M.getOrInsertFunction("bemps_free", ETy);

  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for gridX
  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for gridY
  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for gridZ
  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for blockX
  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for blockY
  ParamTys.push_back(Type::getInt32Ty(Ctx));  // for blockZ
  ParamTys.push_back(Type::getInt64Ty(Ctx));  // for cuda memory size
  FunctionType *BTy = FunctionType::get(Type::getVoidTy(Ctx), ParamTys, false);
  BeaconBegin = M.getOrInsertFunction("bemps_begin", BTy);

  CUDAInfo.collect(M);
  buildCUDATasks(M);
  instrument(M);

  return true;
}

///==================== CUDA Task Builder ============================///

void GPUBeaconPass::buildCUDATasks(Module &M) {
  std::vector<CUDAUnitTask> Tmp;
  // firstly, construct a unitTask for each CUDA kernel, then merge them
  // together if necessary.
  for (auto Invoke : CUDAInfo.getKernelInvokes()) {
    auto GridInfo = getGridCtor(Invoke);
    auto BlockInfo = getBlockCtor(Invoke);
    auto AllocOps = getMemAllocOps(Invoke);
    auto FreeOps = getMemFreeOps(Invoke);
    CUDAUnitTask CT(GridInfo, BlockInfo, Invoke.getPush(), AllocOps, FreeOps);
    Tmp.push_back(CT);
  }

  // merge several unit tasks into a complex task if they
  // share some memory objects
  std::vector<bool> mask(Tmp.size(), false);
  for (int i = 0; i < Tmp.size(); i++) {
    if (mask[i]) continue;
    std::vector<int> Union;
    Union.push_back(i);
    mask[i] = true;

    for (int j = i + 1; j < Tmp.size(); j++) {
      if (mask[j]) continue;
      for (int k : Union) {
        if (needTobeMerged(Tmp[j], Tmp[k])) {
          Union.push_back(j);
          mask[j] = true;
          break;
        }
      }
    }

    if (Union.size() == 1)
      Tasks.push_back(new CUDAUnitTask(Tmp[i]));
    else {
      std::vector<CUDAUnitTask> toBeMerged;
      for (auto l : Union) toBeMerged.push_back(Tmp[l]);
      Tasks.push_back(new CUDAComplexTask(toBeMerged));
    }
  }

  for (auto T : Tasks) T->print();
}

void GPUBeaconPass::instrument(Module &M) {
  IRBuilder<NoFolder> IRB(M.getContext());
  for (auto T : Tasks) {
    auto CUDAMemSizes = T->getCUDAMemSize();
    auto CUDAMemOps = T->getMemAllocOps();
    auto CUDAFreeOps = T->getMemFreeOps();
    auto GridDims = T->getGridDims();
    auto BlockDims = T->getBlockDims();
    auto TID = ConstantInt::getSigned(Type::getInt32Ty(M.getContext()),
                                      T->getTaskID());
    // instrument beacon_begin()
    SmallVector<Value *, 4> Args;
    Args.push_back(TID);
    for (auto D : GridDims) Args.push_back(D);
    for (auto D : BlockDims) Args.push_back(D);

    IRB.SetInsertPoint(*CUDAMemOps.rbegin());

    int n = CUDAMemSizes.size();
    auto tot = CUDAMemSizes[n - 1];
    for (int i = n - 2; i >= 0; i--) {
      tot = IRB.CreateAdd(tot, CUDAMemSizes[i]);
    }

    Args.push_back(tot);

    auto beacon = IRB.CreateCall(BeaconBegin, Args);

    // adjust the postion of cudaMalloc
    for (auto op : CUDAMemOps) {
      if (!postDominate(op, beacon)) {
        // dbgs() << "OP: " << *op;
        for (auto it = op->user_begin(); it != op->user_end(); it++) {
          auto user = dyn_cast<CallInst>(*it);
          if (user && !postDominate(user, beacon)) user->moveAfter(beacon);
          // dbgs() << "\n\tUser: " << *user;
        }
        // dbgs() << "\n\n";
        op->moveAfter(beacon);
      }
    }

    IRB.SetInsertPoint(*CUDAFreeOps.rbegin());
    auto beacon_end = IRB.CreateCall(BeaconRelease, TID);
    for (auto op : CUDAFreeOps) {
      if (!postDominate(beacon_end, op)) {
        beacon_end->moveAfter(op);
      }
    }

    // M.dump();
  }
}

bool GPUBeaconPass::needTobeMerged(CUDATask &A, CUDATask &B) {
  for (auto AA : A.getMemAllocOps()) {
    for (auto BA : B.getMemAllocOps()) {
      if (AA == BA) return true;
    }
  }
  return false;
}

std::set<CallInst *> GPUBeaconPass::getMemAllocOps(InvokeInfo II) {
  std::set<CallInst *> ans;

  for (auto P : II.getMemOperands()) {
    int dist = INT_MAX;
    CallInst *Alloc = nullptr;
    for (auto A : CUDAInfo.getMemAllocs(P)) {
      if (!dominate(A, II)) continue;
      auto tmp = getDistance(A.getCall(), II.getPush());
      if (tmp < dist) {
        dist = tmp;
        Alloc = A.getCall();
      }
    }
    assert(Alloc != nullptr);
    ans.insert(Alloc);
  }

  return ans;
}

std::set<CallInst *> GPUBeaconPass::getMemFreeOps(InvokeInfo II) {
  std::set<CallInst *> ans;

  for (auto P : II.getMemOperands()) {
    DEBUG_WITH_TYPE("build", dbgs() << "Free Pointer: " << *P << "\n");
    int dist = INT_MAX;
    CallInst *FCall = nullptr;
    for (auto F : CUDAInfo.getMemFrees(P)) {
      if (!postDominate(F, II)) continue;
      auto tmp = getDistance(II.getPush(), F.getCall());
      if (tmp < dist) {
        dist = tmp;
        FCall = F.getCall();
      }
    }
    assert(FCall != nullptr);
    ans.insert(FCall);
  }
  return ans;
}

CallInst *GPUBeaconPass::getGridCtor(InvokeInfo II) {
  CallInst *ans;

  auto gridDim = CUDAInfo.getGridDim(II);
  DEBUG_WITH_TYPE("build", dbgs() << "Grid Info: " << *gridDim << "\n");
  for (auto G : CUDAInfo.getDimCtor(gridDim)) {
    if (!dominate(G, II)) continue;
    DEBUG_WITH_TYPE("build", G.print());
    ans = G.getCall();
  }
  return ans;
}

CallInst *GPUBeaconPass::getBlockCtor(InvokeInfo II) {
  CallInst *ans;

  auto blockDim = CUDAInfo.getBlockDim(II);

  DEBUG_WITH_TYPE("build", dbgs() << "\nBlock Info: " << *blockDim << "\n");
  for (auto B : CUDAInfo.getDimCtor(blockDim)) {
    if (!dominate(B, II)) continue;
    DEBUG_WITH_TYPE("build", B.print());
    ans = B.getCall();
  }

  return ans;
}

bool GPUBeaconPass::dominate(MemAllocInfo &MAI, InvokeInfo &II) {
  auto Malloc = MAI.getCall();
  auto Push = II.getPush();
  return dominate(Malloc, Push);
}

bool GPUBeaconPass::dominate(GridCtorInfo &GCI, InvokeInfo &II) {
  auto Ctor = GCI.getCall();
  auto Push = II.getPush();
  return dominate(Ctor, Push);
}
bool GPUBeaconPass::dominate(InvokeInfo &II, MemFreeInfo &MFI) {
  auto Push = II.getPush();
  auto Free = MFI.getCall();
  return dominate(Push, Free);
}

bool GPUBeaconPass::postDominate(MemFreeInfo &MFI, InvokeInfo &II) {
  auto Free = MFI.getCall();
  auto Push = II.getPush();
  bool ans = postDominate(Free, Push);
  return ans;
}

bool GPUBeaconPass::dominate(CallInst *C1, CallInst *C2) {
  Function *F = C1->getFunction();

  // C1 and C2 does not belong to the same function
  if (F != C2->getFunction()) return false;

  auto &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();

  return DT.dominates(C1, C2);
}

bool GPUBeaconPass::postDominate(CallInst *C1, CallInst *C2) {
  Function *F = C1->getFunction();
  // C1 and C2 does not belong to the same function
  if (F != C2->getFunction()) return false;

  PostDominatorTree PDT;
  PDT.recalculate(*F);

  const BasicBlock *BB1 = C1->getParent();
  const BasicBlock *BB2 = C2->getParent();

  if (BB1 != BB2) {
    return PDT.dominates(BB1, BB2);
  } else {
    // Loop through the basic block until we find I1 or I2.
    BasicBlock::const_iterator I = BB1->begin();
    for (; &*I != C1 && &*I != C2; ++I) /*empty*/
      ;
    return &*I == C2;
  }
}

char GPUBeaconPass::ID = 0;

#if 0
static RegisterPass<GPUBeaconPass> X("GB", "GPUBeacon", false, false);

#else

static void registerGPUBeaconPass(const PassManagerBuilder &,
                                  legacy::PassManagerBase &PM) {
  // PM.add(createPromoteMemoryToRegisterPass());
  // PM.add(createLoopSimplifyPass());
  PM.add(new DominatorTreeWrapperPass());
  PM.add(new LoopInfoWrapperPass());
  PM.add(new ScalarEvolutionWrapperPass());
  PM.add(new GPUBeaconPass());
}

// Use EP_OptimizerLast to make sure the pass is run after all other
// optimization passes, such that the debug data is not removed by others
static RegisterStandardPasses RegisterMyPass(
    PassManagerBuilder::EP_OptimizerLast, registerGPUBeaconPass);
#endif