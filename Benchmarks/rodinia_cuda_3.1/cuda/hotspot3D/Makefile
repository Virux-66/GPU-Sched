include ../../common/make.config

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang

INCLUDE := $(CUDA_DIR)/include
FLAGS := -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

SRC = 3D.cu

EXE = 3D 

OUTPUT = *.out

release: $(SRC)
	$(NVCC) --cuda $(SRC) -o /tmp/3D.ii
	$(CC) $(FLAGS) /tmp/3D.ii -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps

enum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
