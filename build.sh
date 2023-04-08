#This script builds or clean all necessary source code, including llvm pass, scheduler, benchmarks and libstatus.

if [ "$1" = "clean" ]
then
    #clean libstatus
    make -C libstatus clean

    #clean llvm pass and scheduler
    rm -rf build

    #clean two benchmarks suits
    make -C Benchmarks_block-level clean
    make -C Benchmarks_kernel-level clean

else
    #build libstatus
    make -C libstatus

    #build llvm pass and scheduler
    mkdir build
    cmake -S src/ -B build/
    cmake --build build/

    #build two benchmarks suits
    make -C Benchmarks_block-level
    make -C Benchmarks_kernel-level
fi