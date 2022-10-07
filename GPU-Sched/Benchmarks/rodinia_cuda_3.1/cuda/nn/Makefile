include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) --cuda nn_cuda.cu -o /tmp/nn_cuda.ii
	clang++ -O1 -S -emit-llvm /tmp/nn_cuda.ii -o /tmp/nn_cuda.ll
	opt --lowerinvoke -load $(WRAPPER_PASS) -WP /tmp/nn_cuda.ll -S -o /tmp/nn_cuda.ll
	clang++ -O1 /tmp/nn_cuda.ll -o nn -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lcudart -lrt -llazy

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
