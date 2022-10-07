#include "Util.h"

std::string getDemangledName(const Function &F) {
  ItaniumPartialDemangler IPD;
  std::string name = F.getName().str();
  if (IPD.partialDemangle(name.c_str())) return name;
  if (IPD.isFunction())
    return IPD.getFunctionBaseName(nullptr, nullptr);
  else return IPD.finishDemangle(nullptr, nullptr);
}

std::string getDemangledName(const Function *F) { return getDemangledName(*F); }

std::string getDemangledName(std::string mangledName) {
  ItaniumPartialDemangler IPD;
  if (IPD.partialDemangle(mangledName.c_str())) return mangledName;

  if (IPD.isFunction())
    return IPD.getFunctionBaseName(nullptr, nullptr);
  else
    return IPD.finishDemangle(nullptr, nullptr);
}

void unreachable(std::string header, Value *V) {
  std::string msg;
  raw_string_ostream rso(msg);
  rso << header << ": " << *V << "\n";
  llvm_unreachable(rso.str().c_str());
}
