#makefile

include ../../common/make.config

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang
FLAGS := -g -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(NVCC) --cuda $< -o /tmp/ex_particle_CUDA_naive_seq.ii
	$(CC) $(FLAGS) /tmp/ex_particle_CUDA_naive_seq.ii -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(NVCC) --cuda $< -o /tmp/ex_particle_CUDA_float_seq.ii
	# $(CC) $(FLAGS) /tmp/ex_particle_CUDA_float_seq.ii -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps -o particlefilter_float
	$(CC) -O1 -S -emit-llvm /tmp/ex_particle_CUDA_float_seq.ii -I$(INCLUDE) -o /tmp/ex.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/ex.ll -S -o /tmp/ex.ll  
	$(CC) -g /tmp/ex.ll -o particlefilter_float -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lm -lrt -llazy
	# $(CC) -g -O1 -S -emit-llvm /tmp/ex_particle_CUDA_float_seq.ii -o /tmp/ex_particle_CUDA_float_seq.ll
	#
	# opt -load $(CHECKER_PASS) -MP /tmp/ex_particle_CUDA_float_seq.ll -o /tmp/ex_particle_CUDA_float_seq.ll
	# $(CC) -g /tmp/ex_particle_CUDA_float_seq.ll -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(WRAPPER_LIB_DIR) -lcudart -lm -lrt -lwrappers -o particlefilter_naive

clean:
	rm particlefilter_naive particlefilter_float
