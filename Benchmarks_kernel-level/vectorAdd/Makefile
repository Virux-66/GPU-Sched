include ../common/make.config

level=3
#C or CPP compiler
CC=/usr/local/bin/clang
CC_FLAGS=-O1
#Common includes and paths for CUDA
INCLUDES=-I../common/inc

#CUDA compiler
NVCC=${CUDA_DIR}/bin/nvcc
NVCC_FLAGS=-I${CUDA_DIR}/include

#make dbg=1 enables NVCC debugging
ifeq (${dbg},1)
    NVCC_FLAGS+=-g -O0
else
    NVCC_FLAGS+=-O1
endif

vectorAdd: vectorAdd.o 
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $<

vectorAdd_native: vectorAdd_native.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -lcudart -lm -o $@ $<

vectorAdd.o: vectorAdd.cu
	${NVCC} ${INCLUDES} --cuda -D LEVEL=${level} -o vectorAdd.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ vectorAdd.ii

vectorAdd_native.o: vectorAdd.cu
	${NVCC} ${INCLUDES} -O1 -c -o $@ $<

clean:
	rm -f vectorAdd vectorAdd.ii vectorAdd.o vectorAdd_native vectorAdd_native.o