#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda_bf16.h>


__global__ void quant_kernel_bf(nv_bfloat16 * x, int * y0,const float c1, const float b1, const int t1, const float c2, const float b2, const int t2, const float c3, const int n, const float v0 ) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;  

    const int shared_mem_size = 8 * sizeof(float) + 2 * sizeof(int);
    extern __shared__ char shared_mem[shared_mem_size];
    float* c1_shared = reinterpret_cast<float*>(shared_mem);
    float* c2_shared = c1_shared + 1;
    float* c3_shared = c2_shared + 1;
    float* b1_shared = c3_shared + 1;
    float* b2_shared = b1_shared + 1;
    float* v0_shared = b2_shared + 1;
    int* t1_shared = reinterpret_cast<int*>(v0_shared + 1);
    int* t2_shared = t1_shared + 1;
    if (threadIdx.x == 0) {
        *c1_shared = c1;
        *c2_shared = c2;
        *c3_shared = c3;
        *b1_shared = b1;
        *b2_shared = b2;
        *t1_shared = t1;
        *t2_shared = t2;
        *v0_shared = v0;
    }
    __syncthreads();     
    
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int tid_c;
    int k;
    float S;
    float v;
    float x_val;
    int y_val=0;
    float r=0.f;
    for (k =0; k< 8; k++){
        tid_c= tid*8+k;
        x_val = __bfloat162float(x[tid_c]);
        S = abs(x_val);
        v = S*499.0f+*v0_shared;
        v -= round(v);
        if(S<*b1_shared){
            x_val = round(x_val*(*c1_shared)+v);
        }
        else if(S<*b2_shared){
            r = round((S - *b1_shared) * (*c2_shared) + v) + *t1_shared;
            x_val = (x_val < 0) ? (-r) : (r);
            }
        else{
            r = round((S - *b2_shared) * (*c3_shared) + v) + *t2_shared;
            x_val = (x_val < 0) ? (-r) : (r);
        }
        if(x_val>7.5){y_val <<= 4; y_val += 1; x[tid_c] = __float2bfloat16(1.0f);}else{y_val <<= 4; y_val |= static_cast<unsigned int>(x_val + 8); x[tid_c] = __float2bfloat16(0.0f);}    
    }
    y[tid] = y_val;
    __syncthreads();
}

template <typename scalar_t>
__global__ void quant_kernel2(scalar_t * x, int * y0,const float c1, const float b1, const int t1, const float c2, const float b2, const int t2, const float c3, const int n, const float v0 ) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;  

    const int shared_mem_size = 8 * sizeof(float) + 2 * sizeof(int);
    extern __shared__ char shared_mem[shared_mem_size];
    float* c1_shared = reinterpret_cast<float*>(shared_mem);
    float* c2_shared = c1_shared + 1;
    float* c3_shared = c2_shared + 1;
    float* b1_shared = c3_shared + 1;
    float* b2_shared = b1_shared + 1;
    float* v0_shared = b2_shared + 1;
    int* t1_shared = reinterpret_cast<int*>(v0_shared + 1);
    int* t2_shared = t1_shared + 1;
    if (threadIdx.x == 0) {
        *c1_shared = c1;
        *c2_shared = c2;
        *c3_shared = c3;
        *b1_shared = b1;
        *b2_shared = b2;
        *t1_shared = t1;
        *t2_shared = t2;
        *v0_shared = v0;
    }
    __syncthreads();     
    
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int tid_c;
    int k;
    float S;
    float v;
    float x_val;
    int y_val=0;
    float r=0.f;
    for (k =0; k< 8; k++){
        tid_c= tid*8+k;
        x_val = x[tid_c];
        S = abs(x_val);
        v = S*499.0f+*v0_shared;
        v -= round(v);
        if(S<*b1_shared){
            x_val = round(x_val*(*c1_shared)+v);
        }
        else if(S<*b2_shared){
            r = round((S - *b1_shared) * (*c2_shared) + v) + *t1_shared;
            x_val = (x_val < 0) ? (-r) : (r);
            }
        else{
            r = round((S - *b2_shared) * (*c3_shared) + v) + *t2_shared;
            x_val = (x_val < 0) ? (-r) : (r);
        }
        if(x_val>7.5){y_val <<= 4; y_val += 1; x[tid_c] = 1;}else{y_val <<= 4; y_val |= static_cast<unsigned int>(x_val + 8); x[tid_c] = 0;}  
    }
    y[tid] = y_val;
    __syncthreads();
}

__global__ void dequant_kernel_bf(const int * y0,nv_bfloat16 * x,const float ic1, const float b1, const int t1, const float ic2, const float b2, const int t2, const float ic3, int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;   

    const int shared_mem_size = 8 * sizeof(float) + 2 * sizeof(int);
    extern __shared__ char shared_mem[shared_mem_size];
    float* ic1_shared = reinterpret_cast<float*>(shared_mem);
    float* ic2_shared = ic1_shared + 1;
    float* ic3_shared = ic2_shared + 1;
    float* b1_shared = ic3_shared + 1;
    float* b2_shared = b1_shared + 1;
    int* t1_shared = reinterpret_cast<int*>(b2_shared + 1);
    int* t2_shared = t1_shared + 1;
    if (threadIdx.x == 0) {
        *ic1_shared = ic1;
        *ic2_shared = ic2;
        *ic3_shared = ic3;
        *b1_shared = b1;
        *b2_shared = b2;
        *t1_shared = t1;
        *t2_shared = t2;
    }
    __syncthreads();
    if (tid >= n) return;

    const unsigned int y = static_cast<unsigned int>(y0[tid]);
    int s;
    int k;
    int tid_c;
    float x_val;
    for (k = 0; k< 8; ++k){
        tid_c = tid*8+k;
        s = (y >> (28-k*4)) &15;
        s = s-8;
        if (s<-*t2_shared) {x_val=(s+*t2_shared)*(*ic3_shared)-*b2_shared;}
        else if(s<-*t1_shared) {x_val=(s+*t1_shared)*(*ic2_shared)-*b1_shared;}
        else if(s<=*t1_shared) {x_val = (s)*(*ic1_shared);}
        else if(s<=*t2_shared) {x_val=(s-*t1_shared)*ic2+*b1_shared;}
        else {x_val=(s-*t2_shared)*(*ic3_shared)+*b2_shared;}
        x[tid_c]= __float2bfloat16(x_val);
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void dequant_kernel2(const int * y0,scalar_t * x,const float ic1, const float b1, const int t1, const float ic2, const float b2, const int t2, const float ic3, const int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;   

    const int shared_mem_size = 8 * sizeof(float) + 2 * sizeof(int);
    extern __shared__ char shared_mem[shared_mem_size];
    float* ic1_shared = reinterpret_cast<float*>(shared_mem);
    float* ic2_shared = ic1_shared + 1;
    float* ic3_shared = ic2_shared + 1;
    float* b1_shared = ic3_shared + 1;
    float* b2_shared = b1_shared + 1;
    int* t1_shared = reinterpret_cast<int*>(b2_shared + 1);
    int* t2_shared = t1_shared + 1;
    if (threadIdx.x == 0) {
        *ic1_shared = ic1;
        *ic2_shared = ic2;
        *ic3_shared = ic3;
        *b1_shared = b1;
        *b2_shared = b2;
        *t1_shared = t1;
        *t2_shared = t2;
    }
    __syncthreads();
    if (tid >= n) return;

    const unsigned int y = static_cast<unsigned int>(y0[tid]);
    int s;
    int k;
    int tid_c;
    scalar_t x_val;
    for (k = 0; k< 8; ++k){
        tid_c = tid*8+k;
        s = (y >> (28-k*4)) &15;
        s = s-8;
        if (s<-*t2_shared) {x_val=(s+*t2_shared)*(*ic3_shared)-*b2_shared;}
        else if(s<-*t1_shared) {x_val=(s+*t1_shared)*(*ic2_shared)-*b1_shared;}
        else if(s<=*t1_shared) {x_val = (s)*(*ic1_shared);}
        else if(s<=*t2_shared) {x_val=(s-*t1_shared)*ic2+*b1_shared;}
        else {x_val=(s-*t2_shared)*(*ic3_shared)+*b2_shared;}
        x[tid_c]=x_val;
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void quant_kernel0(scalar_t * x, int * y0,const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;       
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int tid_c;
    int k;
    float x_val;
    int y_val=0;
    for (k =0; k< 8; k++){
        tid_c= tid*8+k;
        x_val = x[tid_c];
        x_val = round(x_val*8);
        if(x_val>7.5){y_val <<= 4; y_val += 1; x[tid_c] = 1;}else{y_val <<= 4; y_val |= static_cast<unsigned int>(x_val + 8); x[tid_c] = 0;}  
        
    }
    y[tid]=y_val;
    __syncthreads();
}

__global__ void quant_kernel1_bf(nv_bfloat16 * x, int * y0, const int n, const float v0 ) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;       
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int tid_c;
    int k;
    float S;
    float v;
    float x_val;
    int y_val=0;
    for (k =0; k< 8; k++){
        tid_c= tid*8+k;
        x_val = __bfloat162float(x[tid_c]);
        S = abs(x_val);
        v = S*499.0f+v0;
        v = round(v) - v;
        x_val = round(x_val*8+v);
        if(x_val>7.5){y_val <<= 4; y_val += 1; x[tid_c] = __float2bfloat16(1.0f);}else{y_val <<= 4; y_val |= static_cast<unsigned int>(x_val + 8); x[tid_c] = __float2bfloat16(0.0f);}  
        
    }
    y[tid]=y_val;
    __syncthreads();
}

template <typename scalar_t>
__global__ void quant_kernel1(scalar_t * x, int * y0, const int n, const float v0 ) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;       
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int tid_c;
    int k;
    float S;
    float v;
    float x_val;
    int y_val=0;
    for (k =0; k< 8; k++){
        tid_c= tid*8+k;
        x_val = x[tid_c];
        S = abs(x_val);
        v = S*499.0f+v0;
        v = round(v) - v;
        x_val = round(x_val*8+v);
        if(x_val>7.5){y_val <<= 4; y_val += 1; x[tid_c] = 1;}else{y_val <<= 4; y_val |= static_cast<unsigned int>(x_val + 8); x[tid_c] = 0;}  
        
    }
    y[tid]=y_val;
    __syncthreads();
}


__global__ void dequant_kernel1_bf(int * y0,nv_bfloat16 * x, int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;       
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int s;
    int k;
    int tid_c;
    for (k = 0; k< 8; ++k){
        tid_c = tid*8+k;
        s = (y[tid] >> (28-k*4)) &15;
        s = s-8;
        x[tid_c] = __float2bfloat16(s*0.125f);
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void dequant_kernel1(int * y0,scalar_t * x, int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;       
    unsigned int* y = reinterpret_cast<unsigned int*>(y0);

    if (tid >= n) return;

    int s;
    int k;
    int tid_c;
    for (k = 0; k< 8; ++k){
        tid_c = tid*8+k;
        s = (y[tid] >> (28-k*4)) &15;
        s = s-8;
        x[tid_c] = (s)*0.125f;
    }
    __syncthreads();
}


#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

void launch_quant(torch::Tensor& src, torch::Tensor& tgt, const int k1, const float b1, const int k2, const float b2, const int k3, int n) {
    float c1 = k1/b1;
    float c2 = k2/(b2-b1);
    float c3 = k3/(0.5-b2);
    int t1 = k1;
    int t2 = k1+k2;
    float v0 = rand()%1000/1000;
    if (src.dtype() == torch::kBFloat16) {
        quant_kernel_bf<<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<nv_bfloat16*>(src.data_ptr()), tgt.data<int>(),c1,b1,t1,c2,b2,t2,c3,n,v0);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "launch_quant", ([&]{
            quant_kernel2<scalar_t><<<GET_BLOCKS(n), THREADS_PER_BLOCK,
                0,c10::cuda::getCurrentCUDAStream()>>>(src.data<scalar_t>(), tgt.data<int>(),c1,b1,t1,c2,b2,t2,c3,n,v0);
        }));
    }
}

void launch_quant_linear(torch::Tensor& src, torch::Tensor& tgt, int n) {
    float v0 = rand()%1000/1000;
    if (src.dtype() == torch::kBFloat16) {
        quant_kernel1_bf<<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<nv_bfloat16*>(src.data_ptr()), tgt.data<int>(),n,v0);
    } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "launch_quant_linear", ([&]{
        quant_kernel1<scalar_t><<<GET_BLOCKS(n), THREADS_PER_BLOCK,
                0,c10::cuda::getCurrentCUDAStream()>>>(src.data<scalar_t>(), tgt.data<int>(),n,v0);
            }));
    }

}

void launch_quant_linear_s(torch::Tensor& src, torch::Tensor& tgt, int n) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "launch_quant_linear_s", ([&]{
quant_kernel0<scalar_t><<<GET_BLOCKS(n), THREADS_PER_BLOCK,
        0,c10::cuda::getCurrentCUDAStream()>>>(src.data<scalar_t>(), tgt.data<int>(),n);
    }));
}

void launch_dequant(torch::Tensor& src, torch::Tensor& tgt, const int k1, const float b1, const int k2, const float b2, const int k3, int n) {
    float ic1 = b1/k1;
    float ic2 =(b2-b1)/k2;
    float ic3 = (0.5-b2)/k3;
    int t1 = k1;
    int t2 = k1+k2;
    if (tgt.dtype() == torch::kBFloat16) {
        dequant_kernel_bf<<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(src.data<int>(),reinterpret_cast<nv_bfloat16*>(tgt.data_ptr()), ic1,b1,t1,ic2,b2,t2,ic3,n);
    } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(tgt.type(), "launch_dequant", ([&]{
    dequant_kernel2<scalar_t><<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(src.data<int>(), tgt.data<scalar_t>(),ic1,b1,t1,ic2,b2,t2,ic3,n);
        }));
    }
}

void launch_dequant_linear(torch::Tensor& src, torch::Tensor& tgt, int n) {

    if (tgt.dtype() == torch::kBFloat16) {
        dequant_kernel1_bf<<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(src.data<int>(),reinterpret_cast<nv_bfloat16*>(tgt.data_ptr()),n);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(tgt.type(), "launch_dequant_linear", ([&]{
    dequant_kernel1<scalar_t><<<GET_BLOCKS(n), THREADS_PER_BLOCK,
            0,c10::cuda::getCurrentCUDAStream()>>>(src.data<int>(), tgt.data<scalar_t>(),n);
        }));
    }
}
