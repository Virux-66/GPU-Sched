include ../../common/make.config
# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte.out:		main.o
	# nvcc	main.o -I/usr/local/cuda/include -L/usr/local/cuda/lib -lm -lcuda -lcudart -o myocyte.out
	clang main.o -I/usr/local/cuda/include -L$(CUDA_LIB_DIR) -L$(LAZY_LIB_DIR) -lm -lcuda -lcudart -llazy -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	# nvcc	main.cu -c -O3 -g
	nvcc --cuda main.cu -o /tmp/main.ii 
	clang -O1 -S -emit-llvm /tmp/main.ii -o /tmp/main.ll
	opt -load $(WRAPPER_PASS) -WP /tmp/main.ll -S -o /tmp/main.ll
	clang -c /tmp/main.ll -o main.o


# delete all object files
clean:
	rm *.o *.out output.txt
