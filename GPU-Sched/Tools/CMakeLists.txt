cmake_minimum_required(VERSION 3.4)

find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(
	${CLANG_INCLUDE_DIRS}
	)

add_executable(bes-cuda-instrument
  main.cpp
  core.cpp
)

target_link_libraries(bes-cuda-instrument
  PRIVATE
  LLVMDemangle
  LLVMSupport
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangToolingASTDiff
  clangSerialization
  clangAST
  clangAnalysis
)
