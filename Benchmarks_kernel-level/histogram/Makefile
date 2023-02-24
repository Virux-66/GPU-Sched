include ../common/make.config

level=3
#C or CPP compiler
CC=/usr/local/bin/clang++
CC_FLAGS=-O1
#Common includes and path for CUDA
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

histogram: main.o histogram256.o histogram_gold.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $^

histogram_native: main.o histogram256_native.o histogram_gold.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -lcudart -lm -o $@ $^

main.o: main.cpp
	${CC} ${CC_FLAGS} ${INCLUDES} -c -o $@ $<

histogram_gold.o: histogram_gold.cpp
	${CC} ${CC_FLAGS} ${INCLUDES} -c -o $@ $<

histogram256.o: histogram256.cu
	${NVCC} --cuda -D LEVEL=${level} ${INCLUDES} -o histogram256.ii histogram256.cu
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ histogram256.ii

histogram256_native.o: histogram256.cu
	${NVCC} ${NVCC_FLAGS} ${INCLUDES} -c -o $@ $<

clean:
	rm -f histogram histogram256.ii histogram256.o histogram_gold.o main.o histogram256_native.o histogram_native
