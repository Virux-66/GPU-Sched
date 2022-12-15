include ../common/make.config

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

BlackScholes: BlackScholes_gold.o BlackScholes.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $^

BlackScholes_native: BlackScholes_gold.o BlackScholes_native.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -lcudart -lm -o $@ $^

BlackScholes_gold.o: BlackScholes_gold.cpp
	${CC} ${CC_FLAGS} -c -o $@ $<

BlackScholes.o: BlackScholes.cu
	${NVCC} --cuda -o BlackScholes.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ BlackScholes.ii

BlackScholes_native.o: BlackScholes.cu
	${NVCC} -c -o $@ $<

clean:
	rm -f BlackScholes.ii BlackScholes_gold.o BlackScholes.o BlackScholes BlackScholes_native.o BlackScholes_native