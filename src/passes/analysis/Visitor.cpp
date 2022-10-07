
#include "Visitor.h"

#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

void VisitorBase::visitModule(Module &M) {
  Module::FunctionListType &Funcs = M.getFunctionList();
  for (auto it = Funcs.begin(); it != Funcs.end(); it++) {
    Function &F = *it;
    if (F.isIntrinsic() || F.isDeclaration()) continue;
    visitFunction(F);
  }
}

void VisitorBase::visitFunction(Function &F) {
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (isa<CallInst>(&*I)) visitCallInst(dyn_cast<CallInst>(&*I));
  }
}

bool CUDAVisitor::isDim3Struct(Type *Ty) {
  if (isa<PointerType>(Ty))
    Ty = dyn_cast<PointerType>(Ty)->getPointerElementType();
  if (!isa<StructType>(Ty) || dyn_cast<StructType>(Ty)->isLiteral())
    return false;
  return dyn_cast<StructType>(Ty)->getStructName() == "struct.dim3";
}

void CUDAVisitor::visitCallInst(CallInst *CI) {
  if (!CI) return;
  // dbgs() << "\nvisitCallInst: " << *CI << "\n\n";
  Function *Callee = CI->getCalledFunction();

  // Skip the callee if it is not a function but a value
  if (!Callee) {
    dbgs() << "[Warning] Skip: " << *CI << "\n\n";
    return;
  }

  // analyze the function based on it name
  auto name = getDemangledName(Callee);
  if (name == "dim3") {  // meet an dim3 constructor
    GridCtorInfo GCI(CI);
    GridCtors[GCI.getVar()].push_back(GCI);
    DEBUG_WITH_TYPE("visitor", dbgs() << "\n\n[Info] Meet an dim3 constructor: "
                                      << *GCI.getVar() << "\n\t---->" << *CI
                                      << "\n\n");
  } else if (name == "__cudaPushCallConfiguration") {
    // An call __cudaPushCallConfiguration indicates a call to a kernel
    InvokeInfo II(CI);
    KernelInvokes.push_back(II);
    DEBUG_WITH_TYPE("visitor", {
      dbgs() << "[Info] Meet an Kernel Invoke: \n";
      II.print();
    });
  } else if (name == "cudaMalloc") {
    MemAllocInfo MAI(CI);
    MemAllocs[MAI.getObj()].push_back(MAI);
    DEBUG_WITH_TYPE("visitor", {
      dbgs() << "[Info] Meet an CUDA Memory Alloc Operation: \n";
      MAI.print();
    });
  } else if (name == "cudaFree") {
    MemFreeInfo MFI(CI);
    MemFrees[MFI.getObj()].push_back(MFI);
    DEBUG_WITH_TYPE("visitor", {
      dbgs() << "[Info] Meet an CUDA Memory Free Operation: \n";
      MFI.print();
    });
  } else if (auto memcpy = dyn_cast<MemCpyInst>(CI)) {
    auto target = memcpy->getArgOperand(0);
    auto src = memcpy->getArgOperand(1);
    Type *TTy, *STy;
    if (auto BI = dyn_cast<BitCastInst>(target)) {
      TTy = BI->getSrcTy();
      target = BI->getOperand(0);
    }

    if (auto BI = dyn_cast<BitCastInst>(src)) {
      STy = BI->getSrcTy();
      src = BI->getOperand(0);
    }

    if (TTy && STy) {
      if (isDim3Struct(STy) && TTy == STy) AggMap[target] = src;
      if (isDim3Struct(STy) && TTy != STy) CoerseMap[target] = src;
    }

  } else if (!Callee->isIntrinsic() && StringRef(name).startswith("__cuda") ||
             StringRef(name).startswith("cuda")) {
    dbgs() << "[Info] Unhandled CUDA Operation: " << name << "\n";
  }
}

Value *CUDAVisitor::getGridDim(InvokeInfo &II) {
  auto gridDim = II.getGridDimObjs();
  if (CoerseMap[gridDim] && AggMap[CoerseMap[gridDim]])
    gridDim = AggMap[CoerseMap[gridDim]];
  return gridDim;
}

Value *CUDAVisitor::getBlockDim(InvokeInfo &II) {
  auto blockDim = II.getBlockDimObj();
  if (CoerseMap[blockDim] && AggMap[CoerseMap[blockDim]])
    blockDim = AggMap[CoerseMap[blockDim]];
  return blockDim;
}

void CUDAVisitor::print() {
  dbgs() << "\n\nKernel Invokes: \n";
  for (auto KI : KernelInvokes) KI.print();

  dbgs() << "\nDim3 Constructor: \n";
  for (auto GC : GridCtors) {
    for (auto C : GC.second) C.print();
  }
}