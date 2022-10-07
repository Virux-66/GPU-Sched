#include "core.h"

string getDemangledName(FunctionDecl *Func) {
  ItaniumPartialDemangler IPD;
  string name = Func->getNameAsString();
  return name;
  // return demangle(name);
#if 0
  IPD.partialDemangle(name.c_str());
  if (IPD.isFunction())
    return IPD.getFunctionBaseName(nullptr, nullptr);
  else
    return IPD.finishDemangle(nullptr, nullptr);
#endif
}

string getDemangledName(string mangledName) {
  ItaniumPartialDemangler IPD;
  IPD.partialDemangle(mangledName.c_str());
  if (IPD.isFunction())
    return IPD.getFunctionBaseName(nullptr, nullptr);
  else
    return IPD.finishDemangle(nullptr, nullptr);
}

bool PreVisitor::isCUDAKernelInvokeExpr(Stmt *St) {
  // we are insterested in ConditionalOperator because every
  // CUDA kernel (wrapper) call is wrappered as an ConditionalOperator
  auto CO = dyn_cast<ConditionalOperator>(St);
  if (!CO) return false;

  auto Cond = CO->getCond();

  // dereference the ImplicitCastExpr and ParenExpr
  // surrounding the __cudaPushCallConfiguration calls
  if (auto Cast = dyn_cast<CastExpr>(Cond)) Cond = Cast->getSubExpr();
  if (auto PE = dyn_cast<ParenExpr>(Cond)) Cond = PE->getSubExpr();

  // finally the expected stmt should be an CallExpr for
  // __cudaPushCallConfiguration
  if (!isa<CallExpr>(Cond)) return false;

  FunctionDecl *Callee = dyn_cast<CallExpr>(Cond)->getDirectCallee();
  if (!Callee) return false;

  return getDemangledName(Callee) == "__cudaPushCallConfiguration";
}

bool PreVisitor::isCUDAMemoryOperationExpr(Stmt *St) {
  auto Call = dyn_cast<CallExpr>(St);
  if (!Call) return false;
  FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee) return false;
  auto name = getDemangledName(Callee);
  return isCUDAMemoryAllocExpr(name) || isCUDAMemoryFreeExpr(name);
}

bool PreVisitor::isCUDAMemoryAllocExpr(Stmt *St) {
  auto Call = dyn_cast<CallExpr>(St);
  if (!Call) return false;
  FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee) return false;
  auto name = getDemangledName(Callee);
  return isCUDAMemoryAllocExpr(name);
}

bool PreVisitor::isCUDAMemoryFreeExpr(Stmt *St) {
  auto Call = dyn_cast<CallExpr>(St);
  if (!Call) return false;
  FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee) return false;
  auto name = getDemangledName(Callee);
  return isCUDAMemoryFreeExpr(name);
}

bool PreVisitor::isCUDAMemoryAllocExpr(string Name) {
  return Name == "cudaMalloc";
}

bool PreVisitor::isCUDAMemoryFreeExpr(string Name) {
  return Name == "cudaFree";
}

std::pair<Expr *, Expr *> PreVisitor::getCUDAKernelDims(Stmt *St) {
  auto CO = dyn_cast<ConditionalOperator>(St);
  auto Cond = CO->getCond();
  if (auto Cast = dyn_cast<CastExpr>(Cond)) Cond = Cast->getSubExpr();
  if (auto PE = dyn_cast<ParenExpr>(Cond)) Cond = PE->getSubExpr();
  auto Call = dyn_cast<CallExpr>(Cond);
  return {Call->getArg(0), Call->getArg(1)};
}

std::vector<Expr *> PreVisitor::getCUDAKernelArgs(Stmt *St) {
  auto CO = dyn_cast<ConditionalOperator>(St);
  auto Kernel = CO->getFalseExpr();
  assert(Kernel);
  auto Call = dyn_cast<CallExpr>(Kernel);
  assert(Call);

  std::vector<Expr *> result;
  for (int i = 0; i < Call->getNumArgs(); i++) {
    auto arg = Call->getArg(i);
    if (arg->getType()->isPointerType()) result.push_back(arg);
  }
  return result;
}

bool PreVisitor::getCUDAMemoryAllocInfo(Stmt *St) {
  auto Call = dyn_cast<CallExpr>(St);
  auto Target = Call->getArg(0);

  while (!isa<DeclRefExpr>(Target)) {
    if (isa<UnaryOperator>(Target))
      Target = dyn_cast<UnaryOperator>(Target)->getSubExpr();
    else if (isa<CastExpr>(Target))
      Target = dyn_cast<CastExpr>(Target)->getSubExpr();
    else if (isa<ParenExpr>(Target))
      Target = dyn_cast<ParenExpr>(Target)->getSubExpr();
    else {
      std::string msg;
      raw_string_ostream rso(msg);
      St->dump();
      dbgs() << "\n\n";
      Target->dump(rso);
      msg = "Unsupported target in cudaMalloc: " + rso.str();
      llvm_unreachable(msg.c_str());
    }
  }

  auto Pointer = dyn_cast<DeclRefExpr>(Target)->getDecl();

  dbgs() << "Malloc Target: \n";
  Pointer->dump();
  dbgs() << "----------------------\n\n";
  auto Size = Call->getArg(1);
  MemoryAllocations[Pointer].push_back({St, Pointer, Size});
}

bool PreVisitor::getCUDAMemoryFreeInfo(Stmt *St) {
  auto Call = dyn_cast<CallExpr>(St);
  auto Target = Call->getArg(0);
  while (!isa<DeclRefExpr>(Target)) {
    if (isa<UnaryOperator>(Target))
      Target = dyn_cast<UnaryOperator>(Target)->getSubExpr();
    else if (isa<CastExpr>(Target))
      Target = dyn_cast<CastExpr>(Target)->getSubExpr();
    else if (isa<ParenExpr>(Target))
      Target = dyn_cast<ParenExpr>(Target)->getSubExpr();
    else {
      std::string msg;
      raw_string_ostream rso(msg);
      St->dump();
      dbgs() << "\n\n";
      Target->dump(rso);
      msg = "Unsupported target in cudaFree: " + rso.str();
      llvm_unreachable(msg.c_str());
    }
  }
  auto Pointer = dyn_cast<DeclRefExpr>(Target)->getDecl();
  MemoryFrees[Pointer].push_back(Call);
}

bool PreVisitor::VisitFunctionDecl(FunctionDecl *func) {
  if (func->isMain()) func->dump();
  //  string name = func->getNameAsString();
  //  if (func->isThisDeclarationADefinition()) {
  //    if (name.find("__device_stub_") == 0) {
  //      auto kernel = getDemangledName(name.substr(14));
  //      dbgs() << "Find a CUDA kernel: " << kernel << "\n";
  //      CUDAKernels.push_back(kernel);
  //    }
  //  }
  return true;
}

bool PreVisitor::VisitStmt(Stmt *St) {
  if (isCUDAKernelInvokeExpr(St)) {
    // finally we find an CUDA kernel launch, so we need to:
    // 1. get grid size from __cudaPushCallConfiguration parameters.
    // 2. get involved memory objects from the CUDA kernel wrapper.
    auto Dims = getCUDAKernelDims(St);
    auto MemObjs = getCUDAKernelArgs(St);

    KernelInvokes[St].gridDim = Dims.first;
    KernelInvokes[St].blockDim = Dims.second;
    KernelInvokes[St].Params = MemObjs;

    St->dumpColor();
    dbgs() << "****************************************\n\n";
  } else if (isCUDAMemoryAllocExpr(St)) {
    getCUDAMemoryAllocInfo(St);
    St->dumpColor();
    dbgs() << "****************************************\n\n";
  } else if (isCUDAMemoryFreeExpr(St)) {
    getCUDAMemoryFreeInfo(St);
    St->dumpColor();
    dbgs() << "****************************************\n\n";
  }
  return true;
}

bool CUDAVisitor::isCUDAKernelLaunch(string CalleeName) {
  return find(CUDAKernels.begin(), CUDAKernels.end(), CalleeName) !=
         CUDAKernels.end();
}

bool CUDAVisitor::isCUDAKernelLaunch(CallExpr *CE) {
  if (!CE) return false;
  auto Callee = CE->getDirectCallee();
  if (!Callee) return false;
  string CalleeName = getDemangledName(Callee);
  return isCUDAKernelLaunch(CalleeName);
}

bool CUDAVisitor::isCUDAOperation(CallExpr *CE) {
  if (!CE) return false;
  auto Callee = CE->getDirectCallee();
  if (!Callee) return false;
  string CalleeName = getDemangledName(Callee);
  return CalleeName.find("__cuda") == 0 || CalleeName.find("cuda") == 0;
}

bool CUDAVisitor::VisitStmt(Stmt *St) {
  auto CE = dyn_cast<CallExpr>(St);
  if (!CE) return true;

  if (isCUDAKernelLaunch(CE) || isCUDAOperation(CE)) {
    CallSites.push_back(CE);
  }
  return true;
}