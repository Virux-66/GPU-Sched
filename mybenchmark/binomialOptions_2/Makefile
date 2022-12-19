include ../common/make.config

#C or CPP compiler
CC=/usr/local/bin/clang++
CC_FLAGS=-O1
#Common includes and paths for CUDA
INCLUDES=-I../common/inc

#CUDA compiler
NVCC=${CUDA_DIR}/bin/nvcc
NVCC_FLAGS=-I${CUDA_DIR}/include

# make dbg=1 enables NVCC debugging
ifeq (${dbg},1)
	NVCC_FLAGS+=-g -O0
else 
	NVCC_FLAGS+=-O1
endif


binomialOptions_2: binomialOptions_gold.o binomialOptions_kernel.o binomialOptions.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $^

binomialOptions_2_native: binomialOptions_gold.o binomialOptions_kernel_native.o binomialOptions.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -lcudart -lm -o $@ $^

binomialOptions_gold.o: binomialOptions_gold.cpp
	${CC} ${CC_FLAGS} -c -o $@ $<

binomialOptions.o: binomialOptions.cpp
	${CC} -I${CUDA_DIR}/include ${INCLUDES} ${CC_FLAGS} -c -o $@ $<

binomialOptions_kernel.o: binomialOptions_kernel.cu
	${NVCC} ${INCLUDES} --cuda -o binomialOptions_kernel.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ binomialOptions_kernel.ii

binomialOptions_kernel_native.o: binomialOptions_kernel.cu
	${NVCC} -O1 ${INCLUDES} -c -o $@ $<

clean:
	rm -f binomialOptions_gold.o binomialOptions.o binomialOptions_kernel.ii binomialOptions_kernel.o binomialOptions_2 binomialOptions_kernel_native.o binomialOptions_2_native
