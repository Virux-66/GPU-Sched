all: compile link executable

compile:
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c device.cpp -o device.o
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c system.cpp -o system.o
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c profiler.cpp -o profiler.o

link: compile
	nvcc \
	  -Xcompiler -fPIC --shared \
	  device.o system.o profiler.o \
	  -lnvidia-ml -lcublas -lcurand \
	  -o libstatus.so

executable: link
	nvcc \
	  example.cpp \
	  -L. \
	  -lnvidia-ml -lcublas -lcurand -lstatus \
	  -o example 

executable_no_so:
	nvcc \
	  example.cpp device.cpp system.cpp profiler.cpp \
	  -lnvidia-ml -lcublas -lcurand \
	  -o example 
