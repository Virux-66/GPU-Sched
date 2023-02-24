include ../common/make.config

#specify which level of memory hierarchy
level=3
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

matrixMul_1024_native: matrixMul_native.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -lcudart -lm -o $@ $<

matrixMul.o: matrixMul.cu
	${NVCC}  ${INCLUDES} --cuda -D LEVEL=${level} -o matrixMul.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ matrixMul.ii

matrixMul_native.o: matrixMul.cu
	${NVCC} ${INCLUDES} -O1 -c -o $@ $<

clean:
	rm -f matrixMul.ii matrixMul.o matrixMul_1024 matrixMul_1024_native matrixMul_native.o