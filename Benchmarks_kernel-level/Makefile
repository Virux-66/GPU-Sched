#This Makefile can easily compile and clean all benchmarks in just one command
#For compiling all benchmarks:
#	make
#For cleaning those executables and intermediate files:
#	make clean

all:
	+${MAKE} -C ./binomialOptions_64/
	+${MAKE} -C ./binomialOptions_1024/
	+${MAKE} -C ./BlackScholes/
	+${MAKE} -C ./histogram/
	+${MAKE} -C ./matrixMul_1024/
	+${MAKE} -C ./matrixMul_2048/
	+${MAKE} -C ./scan/
	+${MAKE} -C ./vectorAdd/

clean:
	+${MAKE} clean -C ./binomialOptions_64/
	+${MAKE} clean -C ./binomialOptions_1024/
	+${MAKE} clean -C ./BlackScholes/
	+${MAKE} clean -C ./histogram/
	+${MAKE} clean -C ./matrixMul_1024/
	+${MAKE} clean -C ./matrixMul_2048/
	+${MAKE} clean -C ./scan/
	+${MAKE} clean -C ./vectorAdd/
	
native:
	+${MAKE} binomialOptions_64_native -C ./binomialOptions_64/
	+${MAKE} binomialOptions_1024_native -C ./binomialOptions_1024/
	+${MAKE} BlackScholes_native -C ./BlackScholes/
	+${MAKE} histogram_native -C ./histogram/
	+${MAKE} matrixMul_1024_native -C ./matrixMul_1024/
	+${MAKE} matrixMul_2048_native -C ./matrixMul_2048/
	+${MAKE} scan_native -C ./scan/
	+${MAKE} vectorAdd_native -C ./vectorAdd/
