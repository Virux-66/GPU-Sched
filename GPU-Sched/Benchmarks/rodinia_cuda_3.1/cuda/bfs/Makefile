include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CC) --cuda $(SRC) -o /tmp/bfs.ii
	clang -O1 -S -emit-llvm /tmp/bfs.ii -o /tmp/bfs.ll
	opt --lowerinvoke -load $(WRAPPER_PASS) -WP /tmp/bfs.ll -S -o /tmp/bfs.ll
	clang /tmp/bfs.ll -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lrt -lm -llazy
	# clang -Xclang -load -Xclang $(BEMPS_PASS) /tmp/bfs.ii -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lbemps -lcudart -lrt -lm

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
