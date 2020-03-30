#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cfloat>
#include <cassert>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//===
static void hdlCudaErr(cudaError err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define CUDA_ALLOC_DEV_MEM(devPtr,size) hdlCudaErr(cudaMalloc(devPtr,size), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_H2D(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_D2H(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_D2D(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToDevice), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_FREE_DEV_MEM(devPtr) hdlCudaErr(cudaFree(devPtr), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_DEV_MEMSET(devPtr,value,count) hdlCudaErr(cudaMemset(devPtr,value,count), __FUNCTION__, __FILE__, __LINE__)

typedef thrust::device_system_tag device_memspace;
typedef thrust::host_system_tag host_memspace;


template <typename T, typename M>
struct ptr_type {};

template <typename T>
struct ptr_type<T, host_memspace> {
  typedef T* type;
};
template <typename T>
struct ptr_type<T, device_memspace> {
  typedef thrust::device_ptr<T> type;
};
template <typename T, typename M>
struct vector_type {};

template <typename T>
struct vector_type<T, host_memspace> {
  typedef thrust::host_vector<T> type;
};
template <typename T>
struct vector_type<T, device_memspace> {
  typedef thrust::device_vector<T> type;
};

#endif // CUDA_UTILS_H
