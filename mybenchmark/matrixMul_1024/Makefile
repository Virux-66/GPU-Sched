include ../common/make.config

#C or CPP compiler
CC=/usr/local/bin/clang
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


matrixMul_1024: matrixMul.o 
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $<
matrixMul.o: matrixMul.cu
	${NVCC}  ${INCLUDES} --cuda -o matrixMul.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ matrixMul.ii
clean:
	rm -f matrixMul.ii matrixMul.o matrixMul_1024
