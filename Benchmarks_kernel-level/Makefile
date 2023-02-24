#This Makefile can easily compile and clean all benchmarks in just one command
#For compiling all benchmarks:
#	make
#For cleaning those executables and intermediate files:
#	make clean

level=3

all:
	+${MAKE} level=${level} -C  ./binomialOptions_64/
	+${MAKE} level=${level} -C ./binomialOptions_1024/
	+${MAKE} level=${level} -C  ./BlackScholes/
	+${MAKE} level=${level} -C ./histogram/
	+${MAKE} level=${level} -C ./matrixMul_1024/
	+${MAKE} level=${level} -C ./matrixMul_2048/
	+${MAKE} level=${level} -C ./scan/
	+${MAKE} level=${level} -C ./vectorAdd/

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
