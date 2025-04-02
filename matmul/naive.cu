#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <mma.h>

//SGEMM performs C=αAB+βC at single (=32b) precision
__global__ void sgemm_naive(int M,int N,int K,float alpha,const float *A,const float *B, float beta,float *C){
//compute position in C that this thread is responsible for 
const uint x= blockIdx.x *blockDim.x + threadIdx.x;
const uint y= blockIdx.y * blockDim.y + threadIdx.y;

//'if' condition is necessary for when M or N aren't multiples of 32.
if(x<M && y<N){
    float tmp=0.0;
    for(int i=0;i<K;++i){
        tmp += A[x*K+i] *B[i*N+y];
    }
    //C = alpha *(A@B) +beat *C
    C[x*N +y] = alpha *tmp + beta *C[x*N+y];

}


}



void test_sgemmen_naive(){
    const int M=4096;
    const int N=4096;
    const int K=4096;
    const float alpha=2.0;
    const float beta=3.0;

    float *A,*B,*C;
    float *d_A,*d_B,*d_C;

    A=(float*)malloc(M*K*sizeof(float));
    B=(float*)malloc(K*N*sizeof(float));
    C=(float*)malloc(M*N*sizeof(float));

    cudaMalloc(&d_A,M*K*sizeof(float));
    cudaMalloc(&d_B,K*N*sizeof(float));
    cudaMalloc(&d_C,M*N*sizeof(float));

    for(int i=0;i<M*K;++i){
        A[i]=1.0;
    }
    for(int i=0;i<K*N;++i){
        B[i]=1.0;
    }
    for(int i=0;i<M*N;++i){
        C[i]=0.0;
    }

    cudaMemcpy(d_A,A,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,M*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((M+31)/32,(N+31)/32);

    //create timing events

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_naive<<<grid,block>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
    cudaEventRecord(stop);

    //get timing 
    cudaEventSynchronize(stop);
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds,start,stop);

    cudaMemcpy(C,d_C,M*N*sizeof(float),cudaMemcpyDeviceToHost);

    printf("Time taken for sgemm_naive of %dx%d * %dx%d is %f ms\n",M,K,K,N,milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}


int main(){
    test_sgemmen_naive();
    return 0;
}