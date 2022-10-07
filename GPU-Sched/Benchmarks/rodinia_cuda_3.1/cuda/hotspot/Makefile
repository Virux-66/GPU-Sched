include ../../common/make.config

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang

INCLUDE := $(CUDA_DIR)/include
FLAGS := -O1 -S -emit-llvm

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(NVCC) --cuda $(KERNEL_DIM) $(SRC) -o /tmp/hotspot.ii
	$(CC) $(FLAGS) /tmp/hotspot.ii -o /tmp/hotspot.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/hotspot.ll -S -o /tmp/hotspot.ll
	$(CC) $(KERNEL_DIM) /tmp/hotspot.ll -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lm -lrt -llazy
	# $(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
