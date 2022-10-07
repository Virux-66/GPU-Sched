include ../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall -L$(LAZY_LIB_DIR) -llazy

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(OUTPUT) $(KERNEL_DIM) -cuda main.cu -I./AVI -o /tmp/main.ii
	clang -O1 -S -emit-llvm /tmp/main.ii -o /tmp/main.ll
	opt --lowerinvoke -load $(WRAPPER_PASS) -WP /tmp/main.ll -S -o /tmp/main.ll
	clang /tmp/main.ll -c -fPIE -o main.o
	# nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
