include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

release: $(SRC)
	# $(CC) ${KERNEL_DIM} $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 
	$(CC) $(KERNEL_DIM) --cuda $(SRC) -o /tmp/needle.ii
	clang $(KERNEL_DIM) -O1 -S -emit-llvm /tmp/needle.ii -o /tmp/needle.ll 
	opt -load $(WRAPPER_PASS) -WP /tmp/needle.ll -o /tmp/needle.ll
	clang /tmp/needle.ll -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lrt -llazy

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
