__global__ void add(int *data){
    int idx= blockIdx.x *blockDim.x +threadIdx.x;
    data[idx] *=2;
}