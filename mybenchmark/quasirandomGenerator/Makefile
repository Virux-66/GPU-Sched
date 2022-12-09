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

quasirandomGenerator: quasirandomGenerator.o quasirandomGenerator_gold.o quasirandomGenerator_kernel.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $^

quasirandomGenerator.o: quasirandomGenerator.cpp
	${CC} ${CC_FLAGS} ${INCLUDES}  -c -o $@ $<

quasirandomGenerator_gold.o: quasirandomGenerator_gold.cpp
	${CC} ${CC_FLAGS} ${INCLUDES} -c -o $@ $<

quasirandomGenerator_kernel.o: quasirandomGenerator_kernel.cu
	${NVCC} ${INCLUDES} --cuda -o quasirandomGenerator.ii $<
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ quasirandomGenerator.ii

clean:
	rm -f quasirandomGenerator quasirandomGenerator_gold.o quasirandomGenerator.o quasirandomGenerator_kernel.o quasirandomGenerator.ii