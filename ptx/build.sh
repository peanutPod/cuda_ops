nvcc -o add add.cu -lcuda

#create ptx from cu file
nvcc -ptx add_s.cu -o simple_kernel.ptx