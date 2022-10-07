#include "CUDAInfo.h"

#include <llvm/IR/InstIterator.h>

#include "Util.h"

///==================== Memory Alloc Info ============================///

Value *MemAllocInfo::getTarget() {
  Value *ans = Alloc->getArgOperand(0);
  if (isa<LoadInst>(ans)) ans = dyn_cast<LoadInst>(ans)->getOperand(0);
  if (isa<BitCastInst>(ans)) ans = dyn_cast<BitCastInst>(ans)->getOperand(0);
  return ans;
}

Value *MemAllocInfo::getSize() {
  Function *Callee = Alloc->getCalledFunction();
  auto name = getDemangledName(Callee);

  if (name != "cudaMalloc") {
    unreachable("Unsupported Memory Alloc Function ", Alloc);
  }

  if (Alloc->getNumArgOperands() == 2) {
    return Alloc->getArgOperand(1);
  } else {
    // in some case a memory alloc wrapper was generated,
    // especially if the kernel size is a constant, so need to
    // dive into the wrapper function to get the size info
    for (inst_iterator I = inst_begin(Callee), E = inst_end(Callee); I != E;
         ++I) {
      if (auto CI = dyn_cast<CallInst>(&*I)) {
        auto n = getDemangledName(CI->getCalledFunction());
        if (n == "cudaMalloc") return CI->getArgOperand(1);
      }
    }
    unreachable("No Memory Alloc Function Found: ", Callee);
  }
  return nullptr;
}

void MemAllocInfo::print(int indent) {
  dbgs() << "AllocCall: " << *Alloc << "\n\tsize: " << *Size
         << "\n\tTarget: " << *Target << "\n";
}

///==================== Memory Free Info ============================///
Value *MemFreeInfo::getTarget() {
  Value *ans = Call->getArgOperand(0);
  if (isa<BitCastInst>(ans)) ans = dyn_cast<BitCastInst>(ans)->getOperand(0);
  if (isa<LoadInst>(ans)) ans = dyn_cast<LoadInst>(ans)->getOperand(0);
  if (isa<BitCastInst>(ans)) ans = dyn_cast<BitCastInst>(ans)->getOperand(0);
  return ans;
}

///==================== Grid Setup Info ============================///
Value *GridCtorInfo::getTarget() {
  Value *ans = Ctor->getArgOperand(0);
  if (isa<LoadInst>(ans)) ans = dyn_cast<LoadInst>(ans)->getOperand(0);
  if (isa<BitCastInst>(ans)) ans = dyn_cast<BitCastInst>(ans)->getOperand(0);
  return ans;
}

void GridCtorInfo::print() {
  dbgs() << "Variable: " << *Variable << "\n\t\t----> Ctor: " << *Ctor << "\n";
}

///==================== Kernel Invoke Info ============================///

Value *InvokeInfo::getBase(Value *V) {
  auto tmp = V;
  while (!isa<AllocaInst>(tmp)) {
    if (isa<LoadInst>(tmp))
      tmp = dyn_cast<LoadInst>(tmp)->getPointerOperand();
    else if (isa<BitCastInst>(tmp))
      tmp = dyn_cast<BitCastInst>(tmp)->getOperand(0);
    else if (isa<GetElementPtrInst>(tmp)) {
      // what if the pointer is an member of a struct ?
      if (dyn_cast<GetElementPtrInst>(tmp)->hasAllConstantIndices()) break;
      tmp = dyn_cast<GetElementPtrInst>(tmp)->getPointerOperand();
    } else {
      dbgs() << "unsupported value: " << *tmp << " (" << *V << ")\n";
      llvm_unreachable("see error message in above\n\n");
    }
  }
  return tmp;
}

void InvokeInfo::retrieveDims() {
  gridDim = getBase(cudaPush->getArgOperand(0));
  blockDim = getBase(cudaPush->getArgOperand(2));
}

void InvokeInfo::retrieveInvoke() {
  int idx = 0;
  Instruction *tmp = cudaPush->getNextNonDebugInstruction();

  while (!isa<CallInst>(tmp)) {
    if (isa<BranchInst>(tmp)) {
      tmp = dyn_cast<BranchInst>(tmp)->getSuccessor(idx)->getFirstNonPHIOrDbg();
    } else if (auto Cmp = dyn_cast<CmpInst>(tmp)) {
      idx = 1 - Cmp->isTrueWhenEqual();
      tmp = tmp->getNextNonDebugInstruction();
    } else {
      tmp = tmp->getNextNonDebugInstruction();
    }
  }
  KernelInvoke = dyn_cast<CallInst>(tmp);
}

void InvokeInfo::retrieveMemArgs() {
  if (KernelInvoke == nullptr)
    unreachable("No CUDA Kernel Invoke found", cudaPush);

  for (int i = 0; i < KernelInvoke->getNumArgOperands(); i++) {
    auto arg = KernelInvoke->getArgOperand(i);
    if (!arg->getType()->isPointerTy()) continue;
    auto tmp = getBase(arg);
    MemArgs.push_back(tmp);
  }
}

void InvokeInfo::print() {
  dbgs() << "========== Kernel Invoke Info =============";
  dbgs() << "\nkernel: " << *KernelInvoke;
  dbgs() << "\n  Memory Objs: ";
  for (auto Mem : MemArgs) dbgs() << "\n\t" << *Mem;
  dbgs() << "\n  Grid Setups: "
         << "\n\t" << *gridDim << "\n\t" << *blockDim;
  dbgs() << "\n========== Kernel Invoke End =============";
  dbgs() << "\n\n\n";
}

///========================= CUDA Tasks ============================///

std::vector<Value *> CUDAUnitTask::getCUDAMemSize() {
  std::vector<Value *> sizes;
  for (auto allocCall : MemAllocOps) {
    dbgs() << "CUDA Mem Alloc: " << *allocCall
           << "\n\tSize: " << *(allocCall->getArgOperand(1)) << "\n";
    sizes.push_back(allocCall->getArgOperand(1));
  }
  return sizes;
}

std::vector<Value *> CUDAUnitTask::getGridDims() {
  std::vector<Value *> dims;
  std::vector<std::string> name({"", "gridX", "gridY", "gridZ"});

  for (int i = 1; i < gridCtor->getNumArgOperands(); i++) {
    dbgs() << name[i] << ": " << *gridCtor->getArgOperand(i) << "\n";
    dims.push_back(gridCtor->getArgOperand(i));
  }

  return dims;
}

std::vector<Value *> CUDAUnitTask::getBlockDims() {
  std::vector<Value *> dims;
  std::vector<std::string> name({"", "blockX", "blockY", "blockZ"});

  for (int i = 1; i < blockCtor->getNumArgOperands(); i++) {
    dbgs() << name[i] << ": " << *blockCtor->getArgOperand(i) << "\n";
    dims.push_back(blockCtor->getArgOperand(i));
  }

  return dims;
}

void CUDAUnitTask::print() {
  dbgs() << "\n---------------------CUDA Unit Task---------------------\n";
  dbgs() << "Grid: " << *gridCtor << ",\tBlock: " << *blockCtor;
  dbgs() << "\nKernel: " << *KernelInvok;
  dbgs() << "\nMemory Objects: ";
  for (auto alloc : MemAllocOps) dbgs() << "\n\t" << *alloc;
  dbgs() << "\nMemory Object Frees: ";
  for (auto f : MemFreeOps)
    if (f) dbgs() << "\n\t" << *f << "(" << *f->getArgOperand(0) << ")";
  dbgs() << "\n--------------------------------------------------------\n";
}

std::set<CallInst *> CUDAComplexTask::getMemAllocOps() {
  std::set<CallInst *> MemAllocOps;
  for (auto UT : SubTasks) {
    auto tmp = UT.getMemAllocOps();
    MemAllocOps.insert(tmp.begin(), tmp.end());
  }
  return MemAllocOps;
}

std::set<CallInst *> CUDAComplexTask::getMemFreeOps() {
  std::set<CallInst *> MemFreeOps;
  for (auto UT : SubTasks) {
    auto tmp = UT.getMemFreeOps();
    MemFreeOps.insert(tmp.begin(), tmp.end());
  }
  return MemFreeOps;
}

std::vector<Value *> CUDAComplexTask::getCUDAMemSize() {
  auto MemOps = getMemAllocOps();
  std::vector<Value *> sizes;
  for (auto allocCall : MemOps) sizes.push_back(allocCall->getArgOperand(1));
  return sizes;
}

std::vector<Value *> CUDAComplexTask::getGridDims() {
  // FIXME: temporarily we use the grid info of the first subtask
  // for the whole complex task
  return SubTasks[0].getGridDims();
}

std::vector<Value *> CUDAComplexTask::getBlockDims() {
  // FIXME: temporarily we use the grid info of the first subtask
  // for the whole complex task
  return SubTasks[0].getBlockDims();
}

void CUDAComplexTask::print() {
  dbgs() << "\n====================CUDA Complex Task=====================\n";
  dbgs() << "Num. of Unit Tasks: " << SubTasks.size() << "\n";
  for (auto ST : SubTasks) ST.print();

  dbgs() << "\n\nComplexTask Memory Size: ";
  for (auto s : getCUDAMemSize()) dbgs() << "\n\t" << *s;

  dbgs() << "\n\nGrid Dims: ";
  for (auto ut : SubTasks)
    dbgs() << "\n\t" << *(ut.getGridCtor()) << "\n\t" << *(ut.getBlockCtor())
           << "\n";
  dbgs() << "\n==========================================================\n";
}
