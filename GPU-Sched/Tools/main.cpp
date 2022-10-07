#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

#include "core.h"

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory BesOptionCategory("bes-gpu category");

int main(int argc, const char **argv) {
  // parse the command-line args passed to your code
  CommonOptionsParser op(argc, argv, BesOptionCategory);

  // create a new Clang Tool instance (a LibTooling environment)
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  auto Files = Tool.getSourcePaths();
  for (auto f : Files) dbgs() << "File: " << f << "\n";

  // run the Clang Tool, creating a new FrontendAction (explained below)
  // int result = Tool.run(newFrontendActionFactory<BESFrontendAction>());
  int result = Tool.run(newFrontendActionFactory<BESFrontendAction>().get());

  // errs() << "\nFound " << numFunctions << " functions.\n\n";
  // print out the rewritten source code ("rewriter" is a global var.)
  // rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(errs());
  // return result;
  return true;
}
