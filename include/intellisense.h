#pragma once
#ifdef __INTELLISENSE__

void __syncthreads();
int __syncthreads_and(int predicate);
int __syncthreads_or(int predicate);
__device__  float __fdividef(float, float);
__device__  float __cosf(float);

__device__  double __cos(double);
/*
extern __device__ ​ double  __dadd_rd(double, double);
__device__ ​ double  __dadd_rn(double, double);
__device__ ​ double __dadd_ru(double, double);
__device__ ​ double __dadd_rz(double, double);
__device__ ​ double __ddiv_rd(double, double);
__device__ ​ double __ddiv_rn(double, double);
__device__ ​ double __ddiv_ru(double, double);
__device__ ​ double __ddiv_rz(double, double);
__device__ ​ double __dmul_rd(double, double);
__device__ ​ double __dmul_rn(double, double);
__device__ ​ double __dmul_ru(double, double);
extern __device__ ​ double __dmul_rz(double, double);
__device__ ​ double __drcp_rd(double);
__device__ ​ double __drcp_rn(double);
__device__ ​ double __drcp_ru(double);
__device__ ​ double __drcp_rz(double);
__device__ ​ double __dsqrt_rd(double);
__device__ ​ double __dsqrt_rn(double);
__device__ ​ double __dsqrt_ru(double);
__device__ ​ double __dsqrt_rz(double);
__device__ ​ double __dsub_rd(double, double);
__device__ ​ double __dsub_rn(double, double);
__device__ ​ double __dsub_ru(double, double);
__device__ ​ double __dsub_rz(double, double);
__device__ ​ double __fma_rd(double, double, double);
__device__ ​ double __fma_rn(double, double, double);
__device__ ​ double __fma_ru(double, double, double);
__device__ ​ double __fma_rz(double, double, double);
*/
#endif