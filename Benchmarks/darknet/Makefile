GPU=1
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=1

BEMPS_BUILD_DIR = $(abspath ../../build)
BEMPS_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/bemps
WRAPPER_PASS = $(BEMPS_BUILD_DIR)/passes/wrappers/libWrapperPass.so
LAZY_LIB_DIR = $(BEMPS_BUILD_DIR)/runtime/lazy


ARCH= -gencode arch=compute_70,code=sm_70

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=clang
CPP=clang++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
# OPTS=-Ofast
OPTS=
LDFLAGS= -lm -lpthread -L$(LAZY_LIB_DIR) -llazy
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	# $(CC) $(COMMON) $(CFLAGS) -c $< -o $@
	$(CC) $(COMMON) $(CFLAGS) -S -emit-llvm $< -o $<.ll
	opt --lowerinvoke -load $(WRAPPER_PASS) -WP $<.ll -S -o $<.ll
	$(CC) $(COMMON) $(CFLAGS) -c $<.ll -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	# $(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -cuda -c $< -o $@
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -cuda $< -o $<.ii
	$(CPP) $(COMMON) $(CFLAGS) -S -emit-llvm $<.ii -o $<.ll
	opt --lowerinvoke -load $(WRAPPER_PASS) -WP $<.ll -S -o $<.ll
	$(CPP) $(COMMON) $(CFLAGS) -c $<.ll -o $@


obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/* src/*.ii src/*.ll

