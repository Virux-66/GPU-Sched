include ../../../common/make.config

BEMPS_BUILD_DIR = $(abspath ../../../../../build)
BEMPS_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/bemps
BEMPS_PASS = $(BEMPS_BUILD_DIR)/passes/analysis/libGPUBeaconPass.so

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang

FLAGS := -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

EXE = srad

release: $(SRC)
	$(NVCC) --cuda $(SRC) -o /tmp/srad.ii
	$(CC) $(FLAGS) /tmp/srad.ii -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
