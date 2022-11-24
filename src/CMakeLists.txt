project("GPU-Sched")
cmake_minimum_required(VERSION 3.4)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set(CMAKE_BUILD_TYPE Debug)

#find llvm package
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
#message(STATUS ${LLVM_FOUND})

#find ortools package
find_package(ortools CONFIG REQUIRED)
include_directories(${ortools_INCLUDE_DIRS})
link_directories(${ortools_LIBRARY_DIRS})
#message(STATUS ${ortools_FOUND})

#Both these two line are for finding libstatus
include_directories(${CMAKE_SOURCE_DIR}/../../)
link_directories(${CMAKE_SOURCE_DIR}/../../libstatus/)

add_subdirectory(passes)  # Use your pass name here.
add_subdirectory(runtime)  # Use your pass name here.
