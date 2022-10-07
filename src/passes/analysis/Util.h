#ifndef _UTIL_H_
#define _UTIL_H_

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

std::string getDemangledName(const Function &F);
std::string getDemangledName(const Function *F);
std::string getDemangledName(std::string mangledName);

void unreachable(std::string header, Value *V);

#endif