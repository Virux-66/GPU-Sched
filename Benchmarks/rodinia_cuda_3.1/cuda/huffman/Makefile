include ../../common/make.config


NVCC=nvcc
CC = clang++

CUDA_INCLUDEPATH=/usr/local/cuda/include

# NVCC_OPTS=-O3 -arch=sm_35 -Xcompiler -m64 -g -G
NVCC_OPTS=-g -ccbin clang --cuda

# GCC_OPTS=-O3 -Wall -Wextra -m64
IR_OPTS=-g -O1 -S -emit-llvm

OBJ = main_test_cu.o stats_logger.o vlc_kernel_sm64huff.o scan.o pack_kernels.o cpuencode.o
SRC = main_test_cu.cu
EXE = pavle

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

pavle: $(OBJ) 
	# $(NVCC) $(TESTING) $(CACHECWLUT) $(NVCC_OPTS) $(OBJ) -o $(EXE) 
	$(CC) $(TESTING) $(CACHECWLUT) $(OBJ) -g -o $(EXE) -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lrt -llazy

vlc_kernel_sm64huff.o: vlc_kernel_sm64huff.cu 
	# $(NVCC) $(TESTING) -c vlc_kernel_sm64huff.cu $(NVCC_OPTS)
	$(NVCC) $(NVCC_OPTS) $(TESTING) vlc_kernel_sm64huff.cu -o /tmp/vlc_kernel_sm64huff.ii
	$(CC) $(IR_OPTS) /tmp/vlc_kernel_sm64huff.ii -o /tmp/vlc_kernel_sm64huff.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/vlc_kernel_sm64huff.ll -S -o /tmp/vlc_kernel_sm64huff.ll
	$(CC) -c /tmp/vlc_kernel_sm64huff.ll -o $@

scan.o: scan.cu 
	# $(NVCC) -c scan.cu $(NVCC_OPTS)
	$(NVCC) $(NVCC_OPTS) $(TESTING) scan.cu -o /tmp/scan.ii
	$(CC) $(IR_OPTS) /tmp/scan.ii -o /tmp/scan.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/scan.ll -S -o /tmp/scan.ll
	$(CC) -c /tmp/scan.ll -o $@

#cpuencode.o: cpuencode.cu
#	$(NVCC) -c $(NVCC_OPTS) cpuencode.cu

pack_kernels.o: pack_kernels.cu 
	# $(NVCC) -c pack_kernels.cu $(NVCC_OPTS)
	$(NVCC) $(NVCC_OPTS) $(TESTING) pack_kernels.cu -o /tmp/pack_kernels.ii
	$(CC) $(IR_OPTS) /tmp/pack_kernels.ii -o /tmp/pack_kernels.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/pack_kernels.ll -S -o /tmp/pack_kernels.ll
	$(CC) -c /tmp/pack_kernels.ll -o $@

main_test_cu.o: main_test_cu.cu cutil.h
	# $(NVCC) $(TESTING) -c main_test_cu.cu $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) 
	$(NVCC) $(NVCC_OPTS) $(TESTING) main_test_cu.cu -o /tmp/main_test_cu.ii
	$(CC) $(IR_OPTS) /tmp/main_test_cu.ii -o /tmp/main_test_cu.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/main_test_cu.ll -S -o /tmp/main_test_cu.ll
	$(CC) -c /tmp/main_test_cu.ll -o $@

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(EXE) 
