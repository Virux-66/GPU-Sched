include ../../../common/make.config
BEMPS_BUILD_DIR = $(abspath ../../../../../build)
BEMPS_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/bemps
BEMPS_PASS = $(BEMPS_BUILD_DIR)/passes/analysis/libGPUBeaconPass.so

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang
FLAGS := -g -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

INCLUDE := $(CUDA_DIR)/include

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.cu
	$(NVCC)	--cuda main.cu -o /tmp/srad_main.ii
	$(CC) $(FLAGS)	/tmp/srad_main.ii -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lm -lcuda -lcudart -lbemps -o srad

# delete all object files
clean:
	rm *.o srad
