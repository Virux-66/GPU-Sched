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

scan: main.o scan_gold.o scan.o
	${CC} ${CC_FLAGS} -L${CUDA_LIB_DIR} -L${BEMPS_LIB_DIR} -lcudart -lm -lbemps -o $@ $^

main.o: main.cpp
	${CC} ${CC_FLAGS} -c -o $@ $<

scan_gold.o: scan_gold.cpp
	${CC} ${CC_FLAGS} -I${INCLUDES} -c -o $@ $<

scan.o: scan.cu
	${NVCC} ${INCLUDES} --cuda -o scan.ii scan.cu
	${CC} ${CC_FLAGS} -Xclang -load -Xclang ${BEMPS_PASS} -c -o $@ scan.ii
	
clean:
	rm -f main.o scan_gold.o scan.o scan
