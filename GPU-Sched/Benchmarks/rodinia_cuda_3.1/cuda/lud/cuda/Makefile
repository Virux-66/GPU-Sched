BEMPS_BUILD_DIR = $(abspath ../../../../../build)
BEMPS_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/bemps
BEMPS_PASS = $(BEMPS_BUILD_DIR)/passes/analysis/libGPUBeaconPass.so
WRAPPER_PASS = $(BEMPS_BUILD_DIR)/passes/wrappers/libWrapperPass.so
LAZY_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/lazy

# CC = gcc
CC = clang
NVCC = nvcc

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

# NVCCFLAGS += -I../common -O3 -use_fast_math -arch=sm_13 -lm $(SPACE)
NVCCFLAGS += -I../common -use_fast_math $(SPACE)

# CFLAGS += -I../common -I/usr/include/cuda -O3 -Wall $(SPACE)
CFLAGS += -I../common -I/usr/include/cuda -O1 $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<

%.o:	%.cu 
	# $(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) --cuda $< -o /tmp/$@.ii
	$(CC) -O1 -S -emit-llvm $(KERNEL_DIM) $(DEFS) /tmp/$@.ii -o /tmp/$@.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/$@.ll -S -o /tmp/$@.ll
	$(CC) -O1 -fPIE $(KERNEL_DIM) $(DEFS) -o $@ -c /tmp/$@.ll


$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@  $? -L$(LAZY_LIB_DIR) -llazy -lm -use_fast_math

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
