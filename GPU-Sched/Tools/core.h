
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Analysis/CFG.h>
#include <clang/Analysis/CallGraph.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Demangle/Demangle.h>

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

struct KernelProps {
  Expr *gridDim;
  Expr *blockDim;
  std::vector<Expr *> Params;
};

struct MemObjProps {
  Stmt *stmt;
  Decl *base;
  Expr *size;
};

/// PreVisitor is the first pass to initialize the analysis.
/// It will find all CUDA kernel stubs, which is used to
/// figure out all CallExprs to CUDA kernel wrapper,
/// (stub is the client side impl of CUDA Kernel)
/// initilialize the CallGraph, which contains main
/// functions in the source code (this will help us
/// to filter out function imported by predecessor)
class PreVisitor : public RecursiveASTVisitor<PreVisitor> {
 private:
  ASTContext *Ctx;

  // Tracking information about Kernel Invokations,
  // including, the Invokation itself, grid and block dims
  // as well as parameters
  std::map<Stmt *, KernelProps> KernelInvokes;

  // Tracking Information about memeory allocations
  // Decl reprents the pointer, which stores the base address of memory objects
  // Several cudaMalloc could be associated with the Decl
  std::map<Decl *, std::vector<MemObjProps>> MemoryAllocations;

  // Tracking memory frees
  std::map<Decl *, std::vector<Expr *>> MemoryFrees;

  bool isCUDAKernelInvokeExpr(Stmt *St);
  bool isCUDAMemoryOperationExpr(Stmt *St);

  bool isCUDAMemoryAllocExpr(Stmt *St);
  bool isCUDAMemoryFreeExpr(Stmt *St);
  bool isCUDAMemoryAllocExpr(string Name);
  bool isCUDAMemoryFreeExpr(string Name);

  // if the Stmt is to Invoke a CUDA Kernel, getCUDAKernelDims
  // and getCUDAKernelArgs are used to retrive the expression
  // for kernel size and memory objects referred by the kernel
  std::pair<Expr *, Expr *> getCUDAKernelDims(Stmt *St);
  std::vector<Expr *> getCUDAKernelArgs(Stmt *St);

  // if the Stmt is an CUDA memory operation, e.g., cudaMalloc
  // and cudaFree, getCUDAMemoryOperationInfo is to retrive the
  // expressions for memory operations arguments, e.g., memory
  // target, and size expression for cudaMalloc and so on
  bool getCUDAMemoryAllocInfo(Stmt *St);
  bool getCUDAMemoryFreeInfo(Stmt *St);

 public:
  PreVisitor(CompilerInstance *CI) : Ctx(&(CI->getASTContext())) {}

  virtual bool VisitFunctionDecl(FunctionDecl *func);
  virtual bool VisitStmt(Stmt *St);
};

/// CUDAVisitor is the second pass to find all CUDA operations, including
/// calls to CUDA wrappers
class CUDAVisitor : public RecursiveASTVisitor<CUDAVisitor> {
 private:
  ASTContext *Ctx;
  std::vector<string> &CUDAKernels;

  bool isCUDAKernelLaunch(string CalleeName);
  bool isCUDAKernelLaunch(CallExpr *CE);
  bool isCUDAOperation(CallExpr *CE);

  std::vector<CallExpr *> CallSites;

 public:
  CUDAVisitor(CompilerInstance *CI, std::vector<string> &kernels)
      : Ctx(&(CI->getASTContext())), CUDAKernels(kernels) {}
  virtual bool VisitStmt(Stmt *St);
};

class BESASTConsumer : public ASTConsumer {
 private:
  PreVisitor *Init;
  CUDAVisitor *Cuda;

  std::vector<string> CUDAKernels;

 public:
  BESASTConsumer(CompilerInstance *CI)
      : Init(new PreVisitor(CI)), Cuda(new CUDAVisitor(CI, CUDAKernels)) {}

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    // CG->TraverseDecl(Ctx.getTranslationUnitDecl());
    Init->TraverseDecl(Ctx.getTranslationUnitDecl());
    // dbgs() << "CUDA Kernels: \n";
    // for (auto i : CUDAKernels) dbgs() << i << "\n";
    // dbgs() << "\n\n";
    // Cuda->TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class BESFrontendAction : public ASTFrontendAction {
 public:
  virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                         StringRef file) {
    return std::unique_ptr<ASTConsumer>(new BESASTConsumer(&CI));
  }
};