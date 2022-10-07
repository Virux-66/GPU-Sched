include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

release:
	# $(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 
	$(CC) --cuda $(SRC) -o /tmp/pathfinder.ii
	clang++ -O1 -S -emit-llvm /tmp/pathfinder.ii -o /tmp/pathfinder.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/pathfinder.ll -o /tmp/pathfinder.ll
	clang++ /tmp/pathfinder.ll -o $(EXE) -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lrt -lm -llazy
	

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f pathfinder