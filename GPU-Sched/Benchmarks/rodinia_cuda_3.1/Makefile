include common/make.config

RODINIA_BASE_DIR := $(shell pwd)

CUDA_BIN_DIR := $(RODINIA_BASE_DIR)/bin/

CUDA_DIRS := backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud nn	nw srad streamcluster particlefilter pathfinder mummergpu

all:
	cd cuda/backprop;		make;	cp backprop $(CUDA_BIN_DIR)
	cd cuda/bfs; 			make;	cp bfs $(CUDA_BIN_DIR)
	cd cuda/cfd; 			make;	cp euler3d euler3d_double pre_euler3d pre_euler3d_double $(CUDA_BIN_DIR)
	cd cuda/gaussian;		make;	cp gaussian $(CUDA_BIN_DIR)
	cd cuda/heartwall;  		make;	cp heartwall $(CUDA_BIN_DIR)
	cd cuda/hotspot; 		make;	cp hotspot $(CUDA_BIN_DIR)
	cd cuda/kmeans; 		make;	cp kmeans $(CUDA_BIN_DIR)
	cd cuda/lavaMD;			make;	cp lavaMD $(CUDA_BIN_DIR)
	cd cuda/leukocyte;  		make;	cp CUDA/leukocyte $(CUDA_BIN_DIR)
	cd cuda/lud; 			make;	cp cuda/lud_cuda $(CUDA_BIN_DIR)
	cd cuda/nn;				make;	cp nn $(CUDA_BIN_DIR)
	cd cuda/nw; 			make;	cp needle $(CUDA_BIN_DIR)
	cd cuda/srad/srad_v1; 		make;	cp srad $(CUDA_BIN_DIR)/srad_v1
	cd cuda/srad/srad_v2; 		make;   cp srad $(CUDA_BIN_DIR)/srad_v2
	cd cuda/streamcluster;		make;	cp sc_gpu $(CUDA_BIN_DIR)
	cd cuda/particlefilter;		make;	cp particlefilter_naive particlefilter_float $(CUDA_BIN_DIR)       
	cd cuda/pathfinder;		make;	cp pathfinder $(CUDA_BIN_DIR)
	cd cuda/mummergpu;  		make;	cp bin/mummergpu $(CUDA_BIN_DIR)
	cd cuda/hybridsort;              make;   cp hybridsort $(CUDA_BIN_DIR)
	cd cuda/dwt2d;                   make;   cp dwt2d  $(CUDA_BIN_DIR)
	
clean:
	cd $(CUDA_BIN_DIR); rm -f *
	for dir in $(CUDA_DIRS) ; do cd cuda/$$dir ; make clean ; cd ../.. ; done
