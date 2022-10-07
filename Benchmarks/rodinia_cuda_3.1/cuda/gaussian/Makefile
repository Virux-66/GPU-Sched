include ../../common/make.config

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang
FLAGS := -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(NVCC) --cuda $(SRC) -o /tmp/gaussian.ii
	$(CC) $(FLAGS) /tmp/gaussian.ii -o $(EXE).ll -I$(INCLUDE) -S -emit-llvm 
	$(CC) $(FLAGS) /tmp/gaussian.ii -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps

clean:
	rm gaussian
