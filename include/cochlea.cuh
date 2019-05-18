/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

# include "const.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cufft.h>
#include "helper_functions.h"
#include "helper_cuda.h"
#include "intellisense.h"
#define FULL_MASK 0xffffffff
#define DEBUG_CUDA_BLOCKS  0
#define CUDA_PROTECT_NAN    0
#define DEBUG_NANs          0
#define DEBUG_NAN_TX        254
#define DEBUG_START_SAMPLE_LOOP   0
#define DEBUG_END_SAMPLE_LOOP     0
#define DEBUG_START_OUT_LOOP      0
#define DEBUG_END_OUT_LOOP        1
#define DEBUG_BUG_BL        0
#define DEBUG_BUG_TX        0
#define DEBUG_BUG_TS        (0.0e-1)
#define DEBUG_BUG_TE        (0.000025e-1)


#define POWER_OF_TWO 1
__constant__ float model_constants[MODEL_FLOATS_CONSTANTS_SIZE];
__constant__ int model_constants_integers[MODEL_INTEGERS_CONSTANTS_SIZE];
__constant__ float model_Aihc[SECTIONS*LAMBDA_COUNT];
//__constant__ int model_out_sample_index[MAX_NUMBER_OF_BLOCKS];
//__constant__ int model_end_sample_index[MAX_NUMBER_OF_BLOCKS];
__constant__ long model_constants_longs[MODEL_LONGS_CONSTANTS_SIZE];
//__constant__ float model_max_m1_sp_tolerance[MAX_NUMBER_OF_BLOCKS];
//__constant__ float model_throw_tolerance[MAX_NUMBER_OF_BLOCKS];
//__constant__ float mass_const[MAX_WARPS_PER_BM_BLOCK];
//__constant__ float rmass_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float Rd_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float Sd_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float Qd_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float S_ohcd_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float S_tmd_const[MAX_WARPS_PER_BM_BLOCK];
 //__constant__ float R_tmd_const[MAX_WARPS_PER_BM_BLOCK];

#ifdef __CUDACC__
#define LAUNCHBOUNDS(x,y) __launch_bounds__(x,y)
#else
#define LAUNCHBOUNDS(x,y)
#endif

#ifdef __CUDA_ARCH__
#pragma message( "Last modified on " __TIMESTAMP__ " ,Cuda arch found " __CUDA_ARCH__) 
#if (__CUDA_ARCH__ >= 500)
#define KERNEL_BLOCKS 5
#elif (__CUDA_ARCH__ < 500)
#define KERNEL_BLOCKS 4
#endif
#else
#pragma message( "Last modified on " __TIMESTAMP__ " ,Cuda arch not found ") 
#define KERNEL_BLOCKS 4
#endif

typedef float(*Op)(float, float);
typedef float(*Approxiamtor)(float[FIRST_STAGE_MEMSIZE + 2],int,float);
template<typename T> __inline__ __device__ T addition(T val1, T val2) {
	return (val1 + val2);
}
template<typename T,Op op> __inline__ __device__ T warpReduceSum(T val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = op(val, __shfl_down_sync(FULL_MASK, val, offset));
	return val;
}
template<typename T, Op op,int Start_Lane,int Width> __inline__ __device__ std::enable_if_t<Width<32&& Start_Lane<32,T> warpReduceSum(T val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = op(val, __shfl_down_sync(FULL_MASK, val, offset));
	return val;
}

/**
* @returns (cochlea_data_array[index+1] - cochlea_data_array[index-1])/2dx
*/
template<typename T,int offset> __inline__ __device__ T diffArrayAroundIndex(T cochlea_data_array[FIRST_STAGE_MEMSIZE + 2], int index) {
	return (cochlea_data_array[index - offset] - cochlea_data_array[index + offset]);
}
/**
* @returns (cochlea_data_array[index+1] - cochlea_data_array[index-1])/2dx
*/
__inline__ __device__ float linearDerivateApproximation(float cochlea_data_array[FIRST_STAGE_MEMSIZE + 2], int index, float dx) {
	return (diffArrayAroundIndex<float, 1>(cochlea_data_array, index) / (2 * dx));
}

/**
* @returns (8*(cochlea_data_array[index+1] - cochlea_data_array[index-1]) - (cochlea_data_array[index+2] - cochlea_data_array[index-2]))/(12dx);
*/
__inline__ __device__ float quadraticDerivateApproximation(float cochlea_data_array[FIRST_STAGE_MEMSIZE + 2], int index, float dx) {
	if (index>1&&index<FIRST_STAGE_MEMSIZE) return ((8 * diffArrayAroundIndex<float, 1>(cochlea_data_array, index) - diffArrayAroundIndex<float, 2>(cochlea_data_array, index)) / (12 * dx));
	return linearDerivateApproximation(cochlea_data_array, index, dx);
}

/**
* @returns (8*(cochlea_data_array[index+1] - cochlea_data_array[index-1]) - (cochlea_data_array[index+2] - cochlea_data_array[index-2]))/(12dx);
*/
__inline__ __device__ float cubicDerivateApproximation(float cochlea_data_array[FIRST_STAGE_MEMSIZE + 2], int index, float dx) {
	if (index>2 && index<FIRST_STAGE_MEMSIZE-1) return ((45 * diffArrayAroundIndex<float, 1>(cochlea_data_array, index) -9 * diffArrayAroundIndex<float, 2>(cochlea_data_array, index) + diffArrayAroundIndex<float, 3>(cochlea_data_array, index)) / (60 * dx));
	return quadraticDerivateApproximation(cochlea_data_array, index, dx);
}


Approxiamtor approximators[3] = { linearDerivateApproximation ,quadraticDerivateApproximation ,cubicDerivateApproximation };


template<typename T,Op op> __inline__ __device__ T blockReduceSum(T val) {

	static __shared__ T shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum<T,op>(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

								  //read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum<T,op>(val); //Final reduce within first warp

	return val;
}

template<Op op0, Op op1, Op op2> __inline__ __device__
float3 warpReduceTriple(float3 val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val.x = op0(val.x, __shfl_down_sync(FULL_MASK, val.x, offset));
		val.y = op1(val.y, __shfl_down_sync(FULL_MASK, val.y, offset));
		val.z = op2(val.z, __shfl_down_sync(FULL_MASK, val.z, offset));
	}
	return val;
}

template<Op op0, Op op1, Op op2,int partialWarpSize> __inline__ __device__
float3 partialwarpReduceTriple(float3 val) {
	for (int offset = partialWarpSize; offset > 0; offset /= 2) {
		val.x = op0(val.x, __shfl_down_sync(FULL_MASK,val.x, offset));
		val.y = op1(val.y, __shfl_down_sync(FULL_MASK,val.y, offset));
		val.z = op2(val.z, __shfl_down_sync(FULL_MASK,val.z, offset));
	}
	return val;
}

template<Op op0,Op op1,Op op2> __inline__ __device__ float3 blockReduceTripleAggregators(float3 val) {

	static __shared__ float shared[96]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceTriple<op0, op1, op2>(val);
	if (lane == 0) shared[wid] = val.x; // Write reduced value to shared memory
	if (lane == 0) shared[wid+32] = val.y; // Write reduced value to shared memory
	if (lane == 0) shared[wid+64] = val.z; // Write reduced value to shared memory
	__syncthreads();              // Wait for all partial reductions
	// we will integrate everything into the lower warp for synchronization
								  //read from shared memory only if that warp existed
	val.x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
	val.y = (threadIdx.x < blockDim.x / warpSize) ? shared[lane+32] : 0.0f;
	val.z = (threadIdx.x < blockDim.x / warpSize) ? shared[lane+64] : 0.0f;
	if (wid == 0) val = partialwarpReduceTriple<op0, op1, op2,4>(val); //Final reduce within first warp
	return val;
}
template<int Occupancy>
	// BM calculation kernel, while less finely divided on cochlea property consume less memory
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, Occupancy) TripleAggKernelLauncher(
	 float * __restrict__ input_samples,
	 volatile float *saved_speeds,
	 int * __restrict__ Failed_Converged_Time_Node,
	 int * __restrict__ Failed_Converged_Blocks,
	 //volatile float *saved_speeds_buffer,
	 volatile float *mass,
	 volatile float *rsM, // reciprocal mass for multiplication instead of division
	 volatile float *U,
	 volatile float *L,

	 volatile float *R,
	 volatile float *S,
	 volatile float *Q,

	 volatile float *gamma,
	 volatile float *S_ohc,
	 volatile float *S_tm,
	 volatile float *R_tm,
	 float * __restrict__ gen_model_throw_tolerance,
	 float * __restrict__ gen_model_max_m1_sp_tolerance,
	 int * __restrict__ gen_model_out_sample_index,
	 int * __restrict__ gen_model_end_sample_index,
	 float * __restrict__ convergence_time_measurement,
	 float * __restrict__ convergence_time_measurement_blocks,
	 float * __restrict__ convergence_delta_time_iterations,
	 float * __restrict__ convergence_delta_time_iterations_blocks,
	 float * __restrict__ convergence_jacoby_loops_per_iteration,
	 float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
 ) {

	 int mosi = gen_model_out_sample_index[blockIdx.x];
	 int mesi = gen_model_end_sample_index[blockIdx.x];
	 float mtt = gen_model_throw_tolerance[blockIdx.x];
	 float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	 //__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	 //__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	 //__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	 float rsm_ind;
	 //float l_value;
	 //float u_value;
	 __shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	 //__shared__ float curr_time[WARP_SIZE];
	 //__shared__ float curr_time_step[WARP_SIZE];
	 //__shared__ float half_curr_time_step[WARP_SIZE];
	 __shared__ float sx[SX_SIZE];
	 if (threadIdx.x == 0) {
		 // sx[11] = 0;
		 Failed_Converged_Blocks[blockIdx.x] = 0;
		 // convergence_time_measurement_blocks[blockIdx.x] = 0;
		 // convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		 // convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	 }
	 //__shared__ int common_ints[COMMON_INTS_SIZE];
	 //__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	 float3 acc_sp_fault_ind; // combined acc, sp and fault summary for accumulations
	 acc_sp_fault_ind.x = 0.0f;
	 acc_sp_fault_ind.y = 0.0f;
	 acc_sp_fault_ind.z = 0.0f;
	 //__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	 float P_tm_ind;
	 //__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	 float new_disp_ind;
	 //__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	 float new_speed_ind = 0.0f;
	 //float new_speed_value;
	 // copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	 //__shared__ float common_args_vec[1]; 
	 float curr_time_step_f;
	 float curr_time_f;
	 //float float_base_time_f;  
	 //float next_block_first_sample_f; 
	 float S_bm_ind;
	 float new_ohcp_deriv_ind;
	 float new_ohc_psi_ind;
	 float new_TM_disp;
	 float new_TM_sp;

	 float prev_disp_ind;
	 float prev_speed_ind;
	 float prev_accel_ind;
	 float prev_ohc_psi_ind;
	 float prev_ohcp_deriv_ind;
	 //float prev_OW_displacement_ind;//sx[3]
	 //float prev_OW_acceleration_ind; // sx[7]
	 //float prev_OW_speed_ind;	// sx[5]
	 //float new_OW_displacement_ind; // sx[4]
	 //float new_OW_acceleration_ind;   //sx[2]
	 //float new_OW_speed_ind; // sx[6]
	 float prev_TM_disp_ind;
	 float prev_TM_sp_ind;




	 float new_ohc_pressure_ind;



	 float gamma_ind;

	 float R_ind;

	 //float new_disp_ind;
	 //float new_speed_ind;
	 float new_accel_ind;



	 float Q_ind;
	 //float mass_ind;
	 float reciprocal_mass_ind; // for faster calculations
	 float S_ohc_ind;
	 float S_tm_ind;
	 float R_tm_ind;

	 //int wrap_id = threadIdx.x >> 5;


	 prev_TM_disp_ind = 0.0f;
	 new_TM_disp = 0.0f;
	 prev_TM_sp_ind = 0.0f;
	 new_TM_sp = 0.0f;

	 //prev_OW_displacement_ind = 0.0;	  //sx[3]
	 //new_OW_displacement_ind = 0;			//sx[4]
	 //prev_OW_acceleration_ind = 0;
	 //new_OW_acceleration_ind = 0;
	 //prev_OW_speed_ind = 0;
	 //new_OW_speed_ind = 0;   

	 prev_disp_ind = 0.0f;
	 prev_speed_ind = 0.0f;
	 prev_accel_ind = 0.0f;

	 new_disp_ind = 0.0f;
	 //new_disp_vec[threadIdx.x] = 0.0f;
	 //new_speed_ind = 0; 
	 //new_speed_vec[threadIdx.x] = 0; 
	 //new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	 new_accel_ind = 0.0f;

	 //rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	 rsm_ind = rsM[threadIdx.x];
	 //u_vec[threadIdx.x] = U[threadIdx.x];
	 //l_vec[threadIdx.x] = L[threadIdx.x];
	 //u_value = U[threadIdx.x];
	 //l_value = L[threadIdx.x];
	 pressure_vec[threadIdx.x + 1] = 0.0f;
	 /*
	 if (threadIdx.x==0)
	 {
	 pressure_vec[0] = 0.0;
	 pressure_vec[SECTIONS+1] = 0.0;
	 m_vec[0] = 0.0;
	 m_vec[SECTIONS + 1] = 0.0;
	 u_vec[0] = 0.0;
	 u_vec[SECTIONS + 1] = 0.0;
	 l_vec[0] = 0.0;
	 l_vec[SECTIONS + 1] = 0.0;
	 }
	 */

	 if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	 if (threadIdx.x == 0) {
		 pressure_vec[0] = 0.0;
		 pressure_vec[SECTIONS + 1] = 0.0f;
		 sx[15] = 0;
	 }
	 __syncthreads();
	 prev_ohc_psi_ind = 0.0f;
	 prev_ohcp_deriv_ind = 0.0f;
	 new_ohc_psi_ind = 0.0f;



	 //mass_ind = mass[threadIdx.x]; 
	 reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	 gamma_ind = gamma[threadIdx.x];
	 R_ind = R[threadIdx.x];
	 //S_vec[threadIdx.x] = S[threadIdx.x];
	 S_bm_ind = S[threadIdx.x];
	 Q_ind = Q[threadIdx.x];
	 S_ohc_ind = S_ohc[threadIdx.x];
	 S_tm_ind = S_tm[threadIdx.x];
	 R_tm_ind = R_tm[threadIdx.x];


	 curr_time_step_f = model_constants[19];	   //time_step
												   // time offset calculated by nodes and transfered for  float now
												   //int time_offset = nodes_per_time_block*blockIdx.x;
												   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	 curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
						//float_base_time_f = 0.0f;
						//int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


						//if (threadIdx.x == 0) {
						//int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
						//  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
						// isDecoupled = blockIdx.x%isDecoupled; 
						// isDecoupled = isDecoupled > 0 ? 0 : 1;
						// int preDecoupled
						//  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

						//}
						// if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
						// Main Algorithm
						// will be 1 if the next block will be decoupled from this block
						//int preDecoupled = Decouple_Filter;
						// will be one if this block is decoupled from last block
						//int isDecoupled = Decouple_Filter;



	 int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
																  //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																  //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																  //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																  //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																  //first time to output -> starting after first overlaping period
																  //if (threadIdx.x == 0) {
																  // in case of transient write, start write from the begginning
																  // 
																  //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																  //}


																  //first output time of next block
																  // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																  // removed time_offset +
																  //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																  /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																  next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																  next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																  */
																  //offset for first output sample (in units of samples in output array)
																  //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																  //
																  //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	 int another_loop = 1;


	 //int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	 //int Lipschits_en = 1; // unecessary removed



	 //float sx1 = input_samples[input_sample];
	 //float sx2 = input_samples[input_sample+1];
	 if (threadIdx.x <2) {
		 sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	 }
	 __syncthreads();
	 //float P_TM;
	 //float m_inv = 1.0/M[threadIdx.x];

	 // curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	 // previous test bound curr_time_f<next_block_first_sample_f
	 // (out_sample<nodes_per_time_block)
	 // curr_time_f<next_block_first_sample_f
	 while (input_sample<mesi) {



		 __syncthreads();

		 // if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		 // First Step - make approximation using EULER/MEULER


		 for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			 float new_speed_ref = new_speed_ind; //ind;
			 float new_accel_ref = new_accel_ind;

			 if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			 {
				 out_loop = 0;

				 new_TM_disp = fmaf(prev_TM_sp_ind, curr_time_step_f, prev_TM_disp_ind);

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 //new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 new_speed_ind/*ind*/ = fmaf(prev_accel_ind, curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = fmaf(sx[5], curr_time_step_f, sx[3]);
					 sx[6] = fmaf(sx[7], curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 // OHC:  
				 new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind, curr_time_step_f, prev_ohc_psi_ind);



			 }
			 else		//  TRAPEZOIDAL 
			 {

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				 new_speed_ind = prev_accel_ind + new_accel_ind;
				 new_speed_ind/*ind*/ = fmaf(new_speed_ind, 0.5f*curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // not enough shared mem for trapezoidal 
				 // new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = sx[5] + sx[6];
					 sx[4] = fmaf(sx[4], 0.5f*curr_time_step_f, sx[3]);
					 sx[6] = sx[7] + sx[2];
					 sx[6] = fmaf(sx[6], 0.5f*curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				 new_TM_disp = fmaf(new_TM_disp, 0.5f*curr_time_step_f, prev_TM_disp_ind);

				 // OHC: 
				 new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				 new_ohc_psi_ind = fmaf(new_ohc_psi_ind, 0.5f*curr_time_step_f, prev_ohc_psi_ind);

			 }




			 //_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			 float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			 new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			 // Calc_DeltaL_OHC:
			 // 
			 // if (true == _model._OHC_NL_flag)
			 //	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			 // else
			 //	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			 if (out_loop<CUDA_AUX_TM_LOOPS) {
				 float deltaL_disp;
				 float aux_TM;
				 //if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				 //{
				 //float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				 //float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
				 aux_TM = model_constants[6] * new_ohc_psi_ind;
				 aux_TM = tanhf(aux_TM);
				 deltaL_disp = model_constants[5] * aux_TM;

				 //Calc_TM_speed
				 //aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));

				 //}
				 //else
				 //{
				 //	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				 //Calc_TM_speed
				 //aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //	aux_TM = model_constants[6] * new_ohc_psi_ind;
				 //   aux_TM = aux_TM*aux_TM;

				 //}

				 //Calc_TM_speed	 	
				 // aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				 //aux_TM = model_constants[2] * (aux_TM - 1.0);
				 aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
				 aux_TM = model_constants[2] * aux_TM;


				 // Numerator:
				 //N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				 //			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				 //			+ _model._S_tm*_deltaL_disp );

				 //float N11;
				 //float N22;

				 //float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				 //N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				 //N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				 // N_TM_sp replaced by 	new_TM_sp to converse registers
				 //new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
				 new_TM_sp = fmaf(model_constants[14], new_speed_ind,/*ind*/new_TM_sp);
				 new_TM_sp = new_TM_sp*R_tm_ind;
				 new_TM_sp = new_TM_sp*aux_TM;
				 new_TM_sp = fmaf(S_tm_ind, deltaL_disp, new_TM_sp);
				 //new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
				 // P_TM_vec temporary used here to conserve registers, sorry for the mess
				 // P_TM_vec[threadIdx.x] -> P_tm_ind 
				 //P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 P_tm_ind = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_tm_ind);
				 // Denominator:
				 //D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				 //float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				 // D_TM_Sp will replaced by aux_TM to conserve registers...
				 aux_TM = gamma_ind*aux_TM;
				 aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
				 aux_TM = R_tm_ind*aux_TM;
				 new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				 // Calc_Ptm
				 //_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				 P_tm_ind = S_tm_ind*new_TM_disp;
				 P_tm_ind = fmaf(R_tm_ind, new_TM_sp, P_tm_ind);



			 }
			 // Calc_G   
			 //_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			 // calc R_bm_nl

			 //float R_bm_nl;  replaced by G_ind to converse registers
			 //if (alpha_r == 0.0)
			 //	R_bm_nl = R_ind;
			 //else
			 //float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // G_ind calculation deconstructed from 
			 // (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // float G_ind = new_speed_ind * new_speed_ind;
			 //G_ind = fmaf(model_constants[25], G_ind, 1.0f);
			 //G_ind = R_ind*G_ind;
			 //float G_ind = R_ind;
			 //G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
			 float G_ind = fmaf(R_ind, new_speed_ind/*ind*/, S_bm_ind * new_disp_ind/*ind*/);
			 //G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, Sd_const[wrap_id] * new_disp_vec[threadIdx.x]/*ind*/);
			 //G_ind = fmaf(G_ind, -1.0f, -P_tm_ind);
			 G_ind = -G_ind - P_tm_ind;
			 //G_ind = -1.0f*G_ind;
			 //float dx_pow2 = delta_x * delta_x;


			 float Y_ind;

			 // Calc BC	& Calc_Y   		
			 // _Y						= dx_pow2 * _G * _model._Q;	 
			 // _Y[0]					= _bc;	 
			 // _Y[_model._sections-1]	= 0.0;

			 //float new_sample; // will be sx[8]
			 //float _bc;

			 if (threadIdx.x == 0) {
				 //long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				 // relative position of x in the interval
				 //float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				 //float delta = __fdividef(curr_time_f, model_constants[0]);
				 //float delta = curr_time_f* model_constants[1];
				 sx[9] = curr_time_f* model_constants[1];
				 sx[10] = 1.0f - sx[9];
				 sx[8] = fmaf(sx[1], sx[9], sx[0] * sx[10]);



				 /*
				 if (enable_OW)
				 {
				 _bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				 }
				 else
				 {
				 _bc = delta_x * model_a0 * model_Gme *new_sample;
				 }
				 now I can avoid control
				 however since enable_OW = model_constants_integers[1] is always one lets short the formula
				 Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				 */
				 /**
				 * deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				 */
				 Y_ind = model_constants[32] * sx[6];
				 Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				 Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				 Y_ind = model_constants[26] * Y_ind;
				 // also model_constants[17] * sx[8] is removed since enable_OW is always one
				 //Y_ind = _bc;
			 }
			 else if (threadIdx.x == (blockDim.x - 1)) {
				 Y_ind = 0.0f;
			 }
			 else {
				 Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			 }
			 __syncthreads();
			 // float _bc = new_sample; 



			 // Jacobby -> Y
			 int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			 for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				 __syncthreads();
				 float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 2]);
				 tmp_p = Y_ind - tmp_p;
				 // u_vec  is all 1's so it's removed
				 //l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				 tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				 __syncthreads();




				 pressure_vec[threadIdx.x + 1] = tmp_p;


				 // __threadfence();
			 }

			 __syncthreads();

			 // Calc_BM_acceleration	  
			 //_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			 //new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			 new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			 new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			 //__syncthreads(); 

			 // assuming model_constants_integers[1] && enable_OW always active
			 if (threadIdx.x == 0) {

				 //Calc_OW_Acceleration
				 //_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				 //		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				 sx[9] = model_constants[31] * sx[4];
				 sx[9] = fmaf(model_constants[32], sx[6], sx[9]);
				 sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
				 sx[10] = sx[9] + sx[10];
				 sx[2] = model_constants[7] * sx[10];


			 }


			 // Calc_Ohc_Psi_deriv
			 //_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			 // __syncthreads();
			 /**
			 deconstructing
			 new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			 */
			 new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			 new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			 new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
			 new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
			 __syncthreads();

			 //if (threadIdx.x<3)
			 //{
			 //	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			 //} 


			 //////////////////////////// TESTER //////////////////////////////



			 //float prev_time_step = curr_time_step_f;

			 // Lipschitz condition number
			 //float	Lipschitz;			

			 // find the speed error (compared to the past value)
			 // rempved Lipschits_en due to unecessacity	
			 if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			 {

				 acc_sp_fault_ind.x = fabs(new_accel_ref - new_accel_ind);
				 //acc_diff_ind = blockReduceSum<float, fmaxf >(acc_diff_ind);
				 //sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 acc_sp_fault_ind.y = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 //sp_err_ind = blockReduceSum<float, addition<float>  >(sp_err_ind);
				 //float sp_err_limit = ;
				 // deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				 // fault_vec => fault_ind

				 //fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				 acc_sp_fault_ind.z = acc_sp_fault_ind.y - fmax(mtt, new_speed_ind * curr_time_step_f);
				 //fault_ind = blockReduceSum<float, addition<float>  >(fault_ind);
				 acc_sp_fault_ind = blockReduceTripleAggregators<fmaxf, addition<float>, addition<float> >(acc_sp_fault_ind);
				 // TODO - take only sample into account?
				 __syncthreads();


				 // calculate lipschitz number 
				 if (threadIdx.x<32) {
					 float m1_sp_err = acc_sp_fault_ind.y;	 // shared on single warp not necessary
					 float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									   // Lipschitz calculations (threads 0-32)
					 if (m1_sp_err > mmspt) {
						 Lipschitz = acc_sp_fault_ind.x * curr_time_step_f;
						 Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					 }
					 else
						 Lipschitz = 0.0f;

					 //float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					 //int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					 // use the calculated values to decide


					 if (Lipschitz > 6.0f) //2.0 )
					 {
						 // iteration didn't pass, step size decreased
						 curr_time_step_f *= 0.5f;
						 if (curr_time_step_f<model_constants[21]) {
							 curr_time_step_f = model_constants[21];
						 }

						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul >2
					 else if (acc_sp_fault_ind.z > 0.0f) {
						 // another iteration is needed for this step 
						 another_loop = 1;
						 reset_iteration = 0;
					 } // faults > 0
					 else if (Lipschitz < 0.5f)//(float)(0.25) )
					 {
						 // iteration passed, step size increased 
						 curr_time_step_f *= 2.0f;
						 if (curr_time_step_f>model_constants[22]) {
							 curr_time_step_f = model_constants[22];
						 }
						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul <0.25
					 else {
						 another_loop = 1;
						 reset_iteration = 0;
					 }
					 sx[18] = curr_time_step_f; // broadcast result
					 sx[19] = (float)another_loop;
					 sx[20] = (float)reset_iteration;

				 } // threadIdx.x < 32





			 } // !first_iteration
			 else {

				 if (threadIdx.x < 32) {
					 sx[18] = curr_time_step_f;
					 sx[19] = 1.0f; // another loop 
					 sx[20] = 0.0f; // reset_iteration 
				 }

				 //another_loop = 1;	 
				 //reset_iteration = 0;
			 }


			 __syncthreads();


			 curr_time_step_f = sx[18];
			 another_loop = rint(sx[19]);
			 reset_iteration = rint(sx[20]);

			 /////////////////////////////////////////////////////////



			 out_loop++;
			 // oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			 // out_loop >= _CUDA_OUTERN_LOOPS
			 // convergence has failed
			 if (out_loop >= model_constants_integers[11]) {
				 another_loop = 0;
				 reset_iteration = 1;
				 sx[11] = 1;
			 }
			 //if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		 } // end of outern loop
		 another_loop = 1;
		 //float tdiff = curr_time_f - common_args_vec[0];
		 //tdiff = tdiff > 0 ? tdiff : -tdiff;
		// if (threadIdx.x == 0) {
			 // sx[13] += 1.0f; // counter of iteration;
			 // sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			 // sx[15] += sx[14];
		// }
		 // if overlap time passed or its transient write and I'm actually write it	
		 // replaced curr_time_f >= common_args_vec[0]
		 if (curr_time_f > model_constants[20]) {
			 // TODO
			 curr_time_f -= model_constants[20];
			 if (threadIdx.x == 0) {
				 sx[0] = sx[1];
				 sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			 }
			 __syncthreads();
			 // out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			 if (input_sample + 1 >= mosi) {

				 //int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				 // index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				 // out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				 //int t_ind = ;


				 //saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				 saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				 if (threadIdx.x == 0) {
					 Failed_Converged_Time_Node[input_sample] = sx[11];
					 if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					 // float div_times = __fdividef(sx[12], sx[13]);
					 // convergence_time_measurement[input_sample] = div_times;
					 // convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					 // float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					 // convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					 // convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					 // convergence_delta_time_iterations[input_sample] = sx[13];
					 // convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				 }
				 //__syncthreads();


				 //if (threadIdx.x == 0) {
				 //	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				 //}


				 //out_sample++;


			 }
			 else {
				 saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			 }
			 if (threadIdx.x == 0) {
				 sx[11] = 0;
				 sx[12] = 0;
				 sx[13] = 0;
				 sx[15] = 0;
			 }
			 input_sample++;
			 //__syncthreads(); // NEW NEW 
			 //if (threadIdx.x == 0) {
			 //}
		 } // if write data is needed




		   // copy new to past



		 prev_disp_ind = new_disp_ind; //ind;
		 prev_speed_ind = new_speed_ind/*ind*/;
		 prev_accel_ind = new_accel_ind;
		 prev_ohc_psi_ind = new_ohc_psi_ind;
		 prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		 prev_TM_sp_ind = new_TM_sp;
		 prev_TM_disp_ind = new_TM_disp;

		 // model_constants_integers[1] && enable_OW is always 1
		 if (threadIdx.x == 0) {
			 sx[3] = sx[4];
			 sx[5] = sx[6];
			 sx[7] = sx[2];
		 }

		 __syncthreads();
		 curr_time_f += curr_time_step_f;




	 } // end of sample loop

	   // store data in global mem

	   // TODO - signal finished section
	   // wait for new section to be ready (jumps of ...)



	 __syncthreads();
 }
 // BM calculation kernel, while less finely divided on cochlea property consume less memory
 __global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 5) BMOHC_FAST_kernel(
	 float * __restrict__ input_samples,
	 volatile float *saved_speeds,
	 int * __restrict__ Failed_Converged_Time_Node,
	 int * __restrict__ Failed_Converged_Blocks,
	 //volatile float *saved_speeds_buffer,
	 volatile float *mass,
	 volatile float *rsM, // reciprocal mass for multiplication instead of division
	 volatile float *U,
	 volatile float *L,

	 volatile float *R,
	 volatile float *S,
	 volatile float *Q,

	 volatile float *gamma,
	 volatile float *S_ohc,
	 volatile float *S_tm,
	 volatile float *R_tm,
	 float * __restrict__ gen_model_throw_tolerance,
	 float * __restrict__ gen_model_max_m1_sp_tolerance,
	 int * __restrict__ gen_model_out_sample_index,
	 int * __restrict__ gen_model_end_sample_index,
	 float * __restrict__ convergence_time_measurement,
	 float * __restrict__ convergence_time_measurement_blocks,
	 float * __restrict__ convergence_delta_time_iterations,
	 float * __restrict__ convergence_delta_time_iterations_blocks,
	 float * __restrict__ convergence_jacoby_loops_per_iteration,
	 float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
	 ) {

	 int mosi = gen_model_out_sample_index[blockIdx.x];
	 int mesi = gen_model_end_sample_index[blockIdx.x];
	 float mtt = gen_model_throw_tolerance[blockIdx.x];
	 float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	 //__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	 //__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	 //__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	 float rsm_ind;
	 //float l_value;
	 //float u_value;
	 __shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	 //__shared__ float curr_time[WARP_SIZE];
	 //__shared__ float curr_time_step[WARP_SIZE];
	 //__shared__ float half_curr_time_step[WARP_SIZE];
	 __shared__ float sx[SX_SIZE];
	 if (threadIdx.x == 0) { 
		// sx[11] = 0;
		 Failed_Converged_Blocks[blockIdx.x] = 0;
		// convergence_time_measurement_blocks[blockIdx.x] = 0;
		// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	 }
	 //__shared__ int common_ints[COMMON_INTS_SIZE];
	 //__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	 __shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	 __shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	 float P_tm_ind;
	 //__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	 float new_disp_ind;
	 //__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	 float new_speed_ind = 0.0f;
	 //float new_speed_value;
	 // copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	 //__shared__ float common_args_vec[1]; 
	 float curr_time_step_f;
	 float curr_time_f;
	 //float float_base_time_f;  
	 //float next_block_first_sample_f; 
	 float S_bm_ind;
	 float new_ohcp_deriv_ind;
	 float new_ohc_psi_ind;
	 float new_TM_disp;
	 float new_TM_sp;

	 float prev_disp_ind;
	 float prev_speed_ind;
	 float prev_accel_ind;
	 float prev_ohc_psi_ind;
	 float prev_ohcp_deriv_ind;
	 //float prev_OW_displacement_ind;//sx[3]
	 //float prev_OW_acceleration_ind; // sx[7]
	 //float prev_OW_speed_ind;	// sx[5]
	 //float new_OW_displacement_ind; // sx[4]
	 //float new_OW_acceleration_ind;   //sx[2]
	 //float new_OW_speed_ind; // sx[6]
	 float prev_TM_disp_ind;
	 float prev_TM_sp_ind;




	 float new_ohc_pressure_ind;



	 float gamma_ind;

	 float R_ind;

	 //float new_disp_ind;
	 //float new_speed_ind;
	 float new_accel_ind;



	 float Q_ind;
	 //float mass_ind;
	 float reciprocal_mass_ind; // for faster calculations
	 float S_ohc_ind;
	 float S_tm_ind;
	 float R_tm_ind;

	 //int wrap_id = threadIdx.x >> 5;


	 prev_TM_disp_ind = 0.0f;
	 new_TM_disp = 0.0f;
	 prev_TM_sp_ind = 0.0f;
	 new_TM_sp = 0.0f;

	 //prev_OW_displacement_ind = 0.0;	  //sx[3]
	 //new_OW_displacement_ind = 0;			//sx[4]
	 //prev_OW_acceleration_ind = 0;
	 //new_OW_acceleration_ind = 0;
	 //prev_OW_speed_ind = 0;
	 //new_OW_speed_ind = 0;   

	 prev_disp_ind = 0.0f;
	 prev_speed_ind = 0.0f;
	 prev_accel_ind = 0.0f;

	 new_disp_ind = 0.0f;
	 //new_disp_vec[threadIdx.x] = 0.0f;
	 //new_speed_ind = 0; 
	 //new_speed_vec[threadIdx.x] = 0; 
	 //new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	 new_accel_ind = 0.0f;

	 //rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	 rsm_ind = rsM[threadIdx.x];
	 //u_vec[threadIdx.x] = U[threadIdx.x];
	 //l_vec[threadIdx.x] = L[threadIdx.x];
	 //u_value = U[threadIdx.x];
	 //l_value = L[threadIdx.x];
	 pressure_vec[threadIdx.x + 1] = 0.0f;
	 /*
	 if (threadIdx.x==0)
	 {
	 pressure_vec[0] = 0.0;
	 pressure_vec[SECTIONS+1] = 0.0;
	 m_vec[0] = 0.0;
	 m_vec[SECTIONS + 1] = 0.0;
	 u_vec[0] = 0.0;
	 u_vec[SECTIONS + 1] = 0.0;
	 l_vec[0] = 0.0;
	 l_vec[SECTIONS + 1] = 0.0;
	 }
	 */

	 if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	 if (threadIdx.x == 0) {
		 pressure_vec[0] = 0.0;
		 pressure_vec[SECTIONS + 1] = 0.0f;
		 sx[15] = 0;
	 }
	 __syncthreads();
	 prev_ohc_psi_ind = 0.0f;
	 prev_ohcp_deriv_ind = 0.0f;
	 new_ohc_psi_ind = 0.0f;



	 //mass_ind = mass[threadIdx.x]; 
	 reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	 gamma_ind = gamma[threadIdx.x];
	 R_ind = R[threadIdx.x];
	 //S_vec[threadIdx.x] = S[threadIdx.x];
	 S_bm_ind = S[threadIdx.x];
	 Q_ind = Q[threadIdx.x];
	 S_ohc_ind = S_ohc[threadIdx.x];
	 S_tm_ind = S_tm[threadIdx.x];
	 R_tm_ind = R_tm[threadIdx.x];


	 curr_time_step_f = model_constants[19];	   //time_step
	 // time offset calculated by nodes and transfered for  float now
	 //int time_offset = nodes_per_time_block*blockIdx.x;
	 //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	 curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
	 //float_base_time_f = 0.0f;
	 //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


	 //if (threadIdx.x == 0) {
	 //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
	 //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
	 // isDecoupled = blockIdx.x%isDecoupled; 
	 // isDecoupled = isDecoupled > 0 ? 0 : 1;
	 // int preDecoupled
	 //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

	 //}
	 // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
	 // Main Algorithm
	 // will be 1 if the next block will be decoupled from this block
	 //int preDecoupled = Decouple_Filter;
	 // will be one if this block is decoupled from last block
	 //int isDecoupled = Decouple_Filter;



	 int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
	 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
	 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
	 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
	 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
	 //first time to output -> starting after first overlaping period
	 //if (threadIdx.x == 0) {
	 // in case of transient write, start write from the begginning
	 // 
	 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
	 //}


	 //first output time of next block
	 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
	 // removed time_offset +
	 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
	 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
	 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
	 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
	 */
	 //offset for first output sample (in units of samples in output array)
	 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
	 //
	 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	 int another_loop = 1;


	 //int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	 //int Lipschits_en = 1; // unecessary removed



	 //float sx1 = input_samples[input_sample];
	 //float sx2 = input_samples[input_sample+1];
	 if (threadIdx.x <2) {
		 sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	 }
	 __syncthreads();
	 //float P_TM;
	 //float m_inv = 1.0/M[threadIdx.x];

	 // curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	 // previous test bound curr_time_f<next_block_first_sample_f
	 // (out_sample<nodes_per_time_block)
	 // curr_time_f<next_block_first_sample_f
	 while (input_sample<mesi) {



		 __syncthreads();

		 // if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		 // First Step - make approximation using EULER/MEULER


		 for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			 float new_speed_ref = new_speed_ind; //ind;
			 float new_accel_ref = new_accel_ind;

			 if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			 {
				 out_loop = 0;

				 new_TM_disp = fmaf(prev_TM_sp_ind, curr_time_step_f, prev_TM_disp_ind);

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 //new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 new_speed_ind/*ind*/ = fmaf(prev_accel_ind, curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = fmaf(sx[5], curr_time_step_f, sx[3]);
					 sx[6] = fmaf(sx[7], curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 // OHC:  
				 new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind, curr_time_step_f, prev_ohc_psi_ind);



			 } else		//  TRAPEZOIDAL 
			 {

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				 new_speed_ind = prev_accel_ind + new_accel_ind;
				 new_speed_ind/*ind*/ = fmaf(new_speed_ind, 0.5f*curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // not enough shared mem for trapezoidal 
				 // new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = sx[5] + sx[6];
					 sx[4] = fmaf(sx[4], 0.5f*curr_time_step_f, sx[3]);
					 sx[6] = sx[7] + sx[2];
					 sx[6] = fmaf(sx[6], 0.5f*curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				 new_TM_disp = fmaf(new_TM_disp, 0.5f*curr_time_step_f, prev_TM_disp_ind);

				 // OHC: 
				 new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				 new_ohc_psi_ind = fmaf(new_ohc_psi_ind, 0.5f*curr_time_step_f, prev_ohc_psi_ind);

			 }




			 //_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			 float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			 new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			 // Calc_DeltaL_OHC:
			 // 
			 // if (true == _model._OHC_NL_flag)
			 //	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			 // else
			 //	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			 if (out_loop<CUDA_AUX_TM_LOOPS) {
				 float deltaL_disp;
				 float aux_TM;
				 //if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				 //{
				 //float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				 //float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
				 aux_TM = model_constants[6] * new_ohc_psi_ind;
				 aux_TM = tanhf(aux_TM);
				 deltaL_disp = model_constants[5] * aux_TM;

				 //Calc_TM_speed
				 //aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));

				 //}
				 //else
				 //{
				 //	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				 //Calc_TM_speed
				 //aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //	aux_TM = model_constants[6] * new_ohc_psi_ind;
				 //   aux_TM = aux_TM*aux_TM;

				 //}

				 //Calc_TM_speed	 	
				 // aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				 //aux_TM = model_constants[2] * (aux_TM - 1.0);
				 aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
				 aux_TM = model_constants[2] * aux_TM;


				 // Numerator:
				 //N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				 //			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				 //			+ _model._S_tm*_deltaL_disp );

				 //float N11;
				 //float N22;

				 //float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				 //N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				 //N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				 // N_TM_sp replaced by 	new_TM_sp to converse registers
				 //new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
				 new_TM_sp = fmaf(model_constants[14], new_speed_ind,/*ind*/new_TM_sp);
				 new_TM_sp = new_TM_sp*R_tm_ind;
				 new_TM_sp = new_TM_sp*aux_TM;
				 new_TM_sp = fmaf(S_tm_ind, deltaL_disp, new_TM_sp);
				 //new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
				 // P_TM_vec temporary used here to conserve registers, sorry for the mess
				 // P_TM_vec[threadIdx.x] -> P_tm_ind 
				 //P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 P_tm_ind = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_tm_ind);
				 // Denominator:
				 //D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				 //float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				 // D_TM_Sp will replaced by aux_TM to conserve registers...
				 aux_TM = gamma_ind*aux_TM;
				 aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
				 aux_TM = R_tm_ind*aux_TM;
				 new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				 // Calc_Ptm
				 //_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				 P_tm_ind = S_tm_ind*new_TM_disp;
				 P_tm_ind = fmaf(R_tm_ind, new_TM_sp, P_tm_ind);



			 }
			 // Calc_G   
			 //_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			 // calc R_bm_nl

			 //float R_bm_nl;  replaced by G_ind to converse registers
			 //if (alpha_r == 0.0)
			 //	R_bm_nl = R_ind;
			 //else
			 //float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // G_ind calculation deconstructed from 
			 // (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			// float G_ind = new_speed_ind * new_speed_ind;
			 //G_ind = fmaf(model_constants[25], G_ind, 1.0f);
			 //G_ind = R_ind*G_ind;
			 //float G_ind = R_ind;
			 //G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
			 float G_ind = fmaf(R_ind, new_speed_ind/*ind*/, S_bm_ind * new_disp_ind/*ind*/);
			 //G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, Sd_const[wrap_id] * new_disp_vec[threadIdx.x]/*ind*/);
			 //G_ind = fmaf(G_ind, -1.0f, -P_tm_ind);
			 G_ind = - G_ind - P_tm_ind;
			 //G_ind = -1.0f*G_ind;
			 //float dx_pow2 = delta_x * delta_x;


			 float Y_ind;

			 // Calc BC	& Calc_Y   		
			 // _Y						= dx_pow2 * _G * _model._Q;	 
			 // _Y[0]					= _bc;	 
			 // _Y[_model._sections-1]	= 0.0;

			 //float new_sample; // will be sx[8]
			 //float _bc;

			 if (threadIdx.x == 0) {
				 //long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				 // relative position of x in the interval
				 //float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				 //float delta = __fdividef(curr_time_f, model_constants[0]);
				 //float delta = curr_time_f* model_constants[1];
				 sx[9] = curr_time_f* model_constants[1];
				 sx[10] = 1.0f - sx[9];
				 sx[8] = fmaf(sx[1], sx[9], sx[0] * sx[10]);



				 /*
				 if (enable_OW)
				 {
				 _bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				 }
				 else
				 {
				 _bc = delta_x * model_a0 * model_Gme *new_sample;
				 }
				 now I can avoid control
				 however since enable_OW = model_constants_integers[1] is always one lets short the formula
				 Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				 */
				 /**
				 * deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				 */
				 Y_ind = model_constants[32] * sx[6];
				 Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				 Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				 Y_ind = model_constants[26] * Y_ind;
				 // also model_constants[17] * sx[8] is removed since enable_OW is always one
				 //Y_ind = _bc;
			 } else if (threadIdx.x == (blockDim.x - 1)) {
				 Y_ind = 0.0f;
			 } else {
				 Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			 }
			 __syncthreads();
			 // float _bc = new_sample; 



			 // Jacobby -> Y
			 int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			 for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				 __syncthreads();
				 float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 2]);
				 tmp_p = Y_ind - tmp_p;
				 // u_vec  is all 1's so it's removed
				 //l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				 tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				 __syncthreads();




				 pressure_vec[threadIdx.x + 1] = tmp_p;


				 // __threadfence();
			 }

			 __syncthreads();

			 // Calc_BM_acceleration	  
			 //_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			 //new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			 new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			 new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			 //__syncthreads(); 

			 // assuming model_constants_integers[1] && enable_OW always active
			 if (threadIdx.x == 0) {

				 //Calc_OW_Acceleration
				 //_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				 //		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				 sx[9] = model_constants[31] * sx[4];
				 sx[9] = fmaf(model_constants[32], sx[6], sx[9]);
				 sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
				 sx[10] = sx[9] + sx[10];
				 sx[2] = model_constants[7] * sx[10];


			 }


			 // Calc_Ohc_Psi_deriv
			 //_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			 /**
			 deconstructing
			 new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			 */
			 new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			 new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			 new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
			 new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
			 __syncthreads();

			 //if (threadIdx.x<3)
			 //{
			 //	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			 //} 


			 //////////////////////////// TESTER //////////////////////////////



			 //float prev_time_step = curr_time_step_f;

			 // Lipschitz condition number
			 //float	Lipschitz;			

			 // find the speed error (compared to the past value)
			 // rempved Lipschits_en due to unecessacity	
			 if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			 {

				 //float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				 acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				 sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 //float sp_err_limit = ;
				 // deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				 // fault_vec => fault_ind
				 float fault_ind = new_speed_ind * curr_time_step_f;
				 fault_ind = fmax(mtt, fault_ind);
				 fault_ind = sp_err_vec[threadIdx.x] - fault_ind;
				 // TODO - take only sample into account?
				 for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					 __syncthreads();
					 if (threadIdx.x<t_i) {
						 sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						 //fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						 acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					 }

				 }

				 int fault_int = __syncthreads_or(fault_ind > 0.0f);


				 // calculate lipschitz number 
				 if (threadIdx.x<32) {
					 float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					 float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
					 // Lipschitz calculations (threads 0-32)
					 if (m1_sp_err > mmspt) {
						 Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						 Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					 } else
						 Lipschitz = 0.0f;

					 //float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					 //int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					 // use the calculated values to decide


					 if (Lipschitz > 6.0f) //2.0 )
					 {
						 // iteration didn't pass, step size decreased
						 curr_time_step_f *= 0.5f;
						 if (curr_time_step_f<model_constants[21]) {
							 curr_time_step_f = model_constants[21];
						 }

						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul >2
					 else if (fault_int > 0) {
						 // another iteration is needed for this step 
						 another_loop = 1;
						 reset_iteration = 0;
					 } // faults > 0
					 else if (Lipschitz < 0.5f)//(float)(0.25) )
					 {
						 // iteration passed, step size increased 
						 curr_time_step_f *= 2.0f;
						 if (curr_time_step_f>model_constants[22]) {
							 curr_time_step_f = model_constants[22];
						 }
						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul <0.25
					 else {
						 another_loop = 1;
						 reset_iteration = 0;
					 }
					 sp_err_vec[0] = curr_time_step_f; // broadcast result
					 sp_err_vec[1] = (float)another_loop;
					 sp_err_vec[2] = (float)reset_iteration;

				 } // threadIdx.x < 32


				 


			 } // !first_iteration
			 else {
				 
				 if (threadIdx.x < 32) {
					 sp_err_vec[0] = curr_time_step_f;
					 sp_err_vec[1] = 1.0f; // another loop 
					 sp_err_vec[2] = 0.0f; // reset_iteration 
				 }
				 
				 //another_loop = 1;	 
				 //reset_iteration = 0;
			 }


			 __syncthreads();


			 curr_time_step_f = sp_err_vec[0];
			 another_loop = rint(sp_err_vec[1]);
			 reset_iteration = rint(sp_err_vec[2]);
			
			 /////////////////////////////////////////////////////////



			 out_loop++;
			 // oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			 // out_loop >= _CUDA_OUTERN_LOOPS
			 // convergence has failed
			 if (out_loop >= model_constants_integers[11]) {
				 another_loop = 0;
				 reset_iteration = 1;
				 sx[11] = 1;
			 }
			 //if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		 } // end of outern loop
		 another_loop = 1;
		 //float tdiff = curr_time_f - common_args_vec[0];
		 //tdiff = tdiff > 0 ? tdiff : -tdiff;
		 if (threadIdx.x == 0) {
			// sx[13] += 1.0f; // counter of iteration;
			// sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			// sx[15] += sx[14];
		 }
		 // if overlap time passed or its transient write and I'm actually write it	
		 // replaced curr_time_f >= common_args_vec[0]
		 if (curr_time_f > model_constants[20]) {
			 // TODO
			 curr_time_f -= model_constants[20];
			 if (threadIdx.x == 0) {
				 sx[0] = sx[1];
				 sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			 }
			 __syncthreads();
			 // out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			 if (input_sample+1 >= mosi) {

				 //int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				 // index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				 // out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				 //int t_ind = ;


				 //saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				 saved_speeds[(input_sample<<LOG_SECTIONS)|threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				 if (threadIdx.x == 0) {
					 Failed_Converged_Time_Node[input_sample] = sx[11];
					 if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					// float div_times = __fdividef(sx[12], sx[13]);
					// convergence_time_measurement[input_sample] = div_times;
					// convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					// float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					// convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					// convergence_delta_time_iterations[input_sample] = sx[13];
					// convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				 }
																								//__syncthreads();


				 //if (threadIdx.x == 0) {
				 //	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				 //}


				 //out_sample++;


			 } else {
				 saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			 }
			 if (threadIdx.x == 0) {
				 sx[11] = 0;
				 sx[12] = 0;
				 sx[13] = 0;
				 sx[15] = 0;
			 }
			 input_sample++;
			 //__syncthreads(); // NEW NEW 
			 //if (threadIdx.x == 0) {
			 //}
		 } // if write data is needed




		 // copy new to past



		 prev_disp_ind = new_disp_ind; //ind;
		 prev_speed_ind = new_speed_ind/*ind*/;
		 prev_accel_ind = new_accel_ind;
		 prev_ohc_psi_ind = new_ohc_psi_ind;
		 prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		 prev_TM_sp_ind = new_TM_sp;
		 prev_TM_disp_ind = new_TM_disp;

		 // model_constants_integers[1] && enable_OW is always 1
		 if (threadIdx.x == 0) {
			 sx[3] = sx[4];
			 sx[5] = sx[6];
			 sx[7] = sx[2];
		 }

		 __syncthreads();
		 curr_time_f += curr_time_step_f;




	 } // end of sample loop

	 // store data in global mem

	 // TODO - signal finished section
	 // wait for new section to be ready (jumps of ...)



	 __syncthreads();
 }

 // BM calculation kernel, while less finely divided on cochlea property consume less memory
 __global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 8) BMOHC_Triple_Aggragation_FAST_kernel(
	 float * __restrict__ input_samples,
	 volatile float *saved_speeds,
	 int * __restrict__ Failed_Converged_Time_Node,
	 int * __restrict__ Failed_Converged_Blocks,
	 //volatile float *saved_speeds_buffer,
	 volatile float *mass,
	 volatile float *rsM, // reciprocal mass for multiplication instead of division
	 volatile float *U,
	 volatile float *L,

	 volatile float *R,
	 volatile float *S,
	 volatile float *Q,

	 volatile float *gamma,
	 volatile float *S_ohc,
	 volatile float *S_tm,
	 volatile float *R_tm,
	 float * __restrict__ gen_model_throw_tolerance,
	 float * __restrict__ gen_model_max_m1_sp_tolerance,
	 int * __restrict__ gen_model_out_sample_index,
	 int * __restrict__ gen_model_end_sample_index,
	 float * __restrict__ convergence_time_measurement,
	 float * __restrict__ convergence_time_measurement_blocks,
	 float * __restrict__ convergence_delta_time_iterations,
	 float * __restrict__ convergence_delta_time_iterations_blocks,
	 float * __restrict__ convergence_jacoby_loops_per_iteration,
	 float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
 ) {

	 int mosi = gen_model_out_sample_index[blockIdx.x];
	 int mesi = gen_model_end_sample_index[blockIdx.x];
	 float mtt = gen_model_throw_tolerance[blockIdx.x];
	 float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	 //__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	 //__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	 //__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	 float rsm_ind;
	 //float l_value;
	 //float u_value;
	 __shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	 //__shared__ float curr_time[WARP_SIZE];
	 //__shared__ float curr_time_step[WARP_SIZE];
	 //__shared__ float half_curr_time_step[WARP_SIZE];
	 __shared__ float sx[SX_SIZE];
	 if (threadIdx.x == 0) {
		 // sx[11] = 0;
		 Failed_Converged_Blocks[blockIdx.x] = 0;
		 // convergence_time_measurement_blocks[blockIdx.x] = 0;
		 // convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		 // convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	 }
	 //__shared__ int common_ints[COMMON_INTS_SIZE];
	 //__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	 float3 acc_sp_fault_ind; // combined acc, sp and fault summary for accumulations
	 acc_sp_fault_ind.x = 0.0f;
	 acc_sp_fault_ind.y = 0.0f;
	 acc_sp_fault_ind.z = 0.0f;
	 //__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	 float P_tm_ind;
	 //__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	 float new_disp_ind;
	 //__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	 float new_speed_ind = 0.0f;
	 //float new_speed_value;
	 // copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	 //__shared__ float common_args_vec[1]; 
	 float curr_time_step_f;
	 float curr_time_f;
	 //float float_base_time_f;  
	 //float next_block_first_sample_f; 
	 float S_bm_ind;
	 float new_ohcp_deriv_ind;
	 float new_ohc_psi_ind;
	 float new_TM_disp;
	 float new_TM_sp;

	 float prev_disp_ind;
	 float prev_speed_ind;
	 float prev_accel_ind;
	 float prev_ohc_psi_ind;
	 float prev_ohcp_deriv_ind;
	 //float prev_OW_displacement_ind;//sx[3]
	 //float prev_OW_acceleration_ind; // sx[7]
	 //float prev_OW_speed_ind;	// sx[5]
	 //float new_OW_displacement_ind; // sx[4]
	 //float new_OW_acceleration_ind;   //sx[2]
	 //float new_OW_speed_ind; // sx[6]
	 float prev_TM_disp_ind;
	 float prev_TM_sp_ind;




	 float new_ohc_pressure_ind;



	 float gamma_ind;

	 float R_ind;

	 //float new_disp_ind;
	 //float new_speed_ind;
	 float new_accel_ind;



	 float Q_ind;
	 //float mass_ind;
	 float reciprocal_mass_ind; // for faster calculations
	 float S_ohc_ind;
	 float S_tm_ind;
	 float R_tm_ind;

	 //int wrap_id = threadIdx.x >> 5;


	 prev_TM_disp_ind = 0.0f;
	 new_TM_disp = 0.0f;
	 prev_TM_sp_ind = 0.0f;
	 new_TM_sp = 0.0f;

	 //prev_OW_displacement_ind = 0.0;	  //sx[3]
	 //new_OW_displacement_ind = 0;			//sx[4]
	 //prev_OW_acceleration_ind = 0;
	 //new_OW_acceleration_ind = 0;
	 //prev_OW_speed_ind = 0;
	 //new_OW_speed_ind = 0;   

	 prev_disp_ind = 0.0f;
	 prev_speed_ind = 0.0f;
	 prev_accel_ind = 0.0f;

	 new_disp_ind = 0.0f;
	 //new_disp_vec[threadIdx.x] = 0.0f;
	 //new_speed_ind = 0; 
	 //new_speed_vec[threadIdx.x] = 0; 
	 //new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	 new_accel_ind = 0.0f;

	 //rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	 rsm_ind = rsM[threadIdx.x];
	 //u_vec[threadIdx.x] = U[threadIdx.x];
	 //l_vec[threadIdx.x] = L[threadIdx.x];
	 //u_value = U[threadIdx.x];
	 //l_value = L[threadIdx.x];
	 pressure_vec[threadIdx.x + 1] = 0.0f;
	 /*
	 if (threadIdx.x==0)
	 {
	 pressure_vec[0] = 0.0;
	 pressure_vec[SECTIONS+1] = 0.0;
	 m_vec[0] = 0.0;
	 m_vec[SECTIONS + 1] = 0.0;
	 u_vec[0] = 0.0;
	 u_vec[SECTIONS + 1] = 0.0;
	 l_vec[0] = 0.0;
	 l_vec[SECTIONS + 1] = 0.0;
	 }
	 */

	 if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	 if (threadIdx.x == 0) {
		 pressure_vec[0] = 0.0;
		 pressure_vec[SECTIONS + 1] = 0.0f;
		 sx[15] = 0;
	 }
	 __syncthreads();
	 prev_ohc_psi_ind = 0.0f;
	 prev_ohcp_deriv_ind = 0.0f;
	 new_ohc_psi_ind = 0.0f;



	 //mass_ind = mass[threadIdx.x]; 
	 reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	 gamma_ind = gamma[threadIdx.x];
	 R_ind = R[threadIdx.x];
	 //S_vec[threadIdx.x] = S[threadIdx.x];
	 S_bm_ind = S[threadIdx.x];
	 Q_ind = Q[threadIdx.x];
	 S_ohc_ind = S_ohc[threadIdx.x];
	 S_tm_ind = S_tm[threadIdx.x];
	 R_tm_ind = R_tm[threadIdx.x];


	 curr_time_step_f = model_constants[19];	   //time_step
												   // time offset calculated by nodes and transfered for  float now
												   //int time_offset = nodes_per_time_block*blockIdx.x;
												   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	 curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
						//float_base_time_f = 0.0f;
						//int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


						//if (threadIdx.x == 0) {
						//int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
						//  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
						// isDecoupled = blockIdx.x%isDecoupled; 
						// isDecoupled = isDecoupled > 0 ? 0 : 1;
						// int preDecoupled
						//  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

						//}
						// if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
						// Main Algorithm
						// will be 1 if the next block will be decoupled from this block
						//int preDecoupled = Decouple_Filter;
						// will be one if this block is decoupled from last block
						//int isDecoupled = Decouple_Filter;



	 int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
																  //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																  //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																  //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																  //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																  //first time to output -> starting after first overlaping period
																  //if (threadIdx.x == 0) {
																  // in case of transient write, start write from the begginning
																  // 
																  //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																  //}


																  //first output time of next block
																  // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																  // removed time_offset +
																  //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																  /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																  next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																  next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																  */
																  //offset for first output sample (in units of samples in output array)
																  //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																  //
																  //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	 int another_loop = 1;


	 //int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	 //int Lipschits_en = 1; // unecessary removed



	 //float sx1 = input_samples[input_sample];
	 //float sx2 = input_samples[input_sample+1];
	 if (threadIdx.x <2) {
		 sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	 }
	 __syncthreads();
	 //float P_TM;
	 //float m_inv = 1.0/M[threadIdx.x];

	 // curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	 // previous test bound curr_time_f<next_block_first_sample_f
	 // (out_sample<nodes_per_time_block)
	 // curr_time_f<next_block_first_sample_f
	 while (input_sample<mesi) {



		 __syncthreads();

		 // if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		 // First Step - make approximation using EULER/MEULER


		 for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			 float new_speed_ref = new_speed_ind; //ind;
			 float new_accel_ref = new_accel_ind;

			 if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			 {
				 out_loop = 0;

				 new_TM_disp = fmaf(prev_TM_sp_ind, curr_time_step_f, prev_TM_disp_ind);

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 //new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 new_speed_ind/*ind*/ = fmaf(prev_accel_ind, curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = fmaf(sx[5], curr_time_step_f, sx[3]);
					 sx[6] = fmaf(sx[7], curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 // OHC:  
				 new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind, curr_time_step_f, prev_ohc_psi_ind);



			 }
			 else		//  TRAPEZOIDAL 
			 {

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				 new_speed_ind = prev_accel_ind + new_accel_ind;
				 new_speed_ind/*ind*/ = fmaf(new_speed_ind, 0.5f*curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				 // not enough shared mem for trapezoidal 
				 // new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = sx[5] + sx[6];
					 sx[4] = fmaf(sx[4], 0.5f*curr_time_step_f, sx[3]);
					 sx[6] = sx[7] + sx[2];
					 sx[6] = fmaf(sx[6], 0.5f*curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				 new_TM_disp = fmaf(new_TM_disp, 0.5f*curr_time_step_f, prev_TM_disp_ind);

				 // OHC: 
				 new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				 new_ohc_psi_ind = fmaf(new_ohc_psi_ind, 0.5f*curr_time_step_f, prev_ohc_psi_ind);

			 }




			 //_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			 float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			 new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			 // Calc_DeltaL_OHC:
			 // 
			 // if (true == _model._OHC_NL_flag)
			 //	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			 // else
			 //	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			 if (out_loop<CUDA_AUX_TM_LOOPS) {
				 float deltaL_disp;
				 float aux_TM;
				 //if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				 //{
				 //float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				 //float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
				 aux_TM = model_constants[6] * new_ohc_psi_ind;
				 aux_TM = tanhf(aux_TM);
				 deltaL_disp = model_constants[5] * aux_TM;

				 //Calc_TM_speed
				 //aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));

				 //}
				 //else
				 //{
				 //	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				 //Calc_TM_speed
				 //aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //	aux_TM = model_constants[6] * new_ohc_psi_ind;
				 //   aux_TM = aux_TM*aux_TM;

				 //}

				 //Calc_TM_speed	 	
				 // aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				 //aux_TM = model_constants[2] * (aux_TM - 1.0);
				 aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
				 aux_TM = model_constants[2] * aux_TM;


				 // Numerator:
				 //N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				 //			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				 //			+ _model._S_tm*_deltaL_disp );

				 //float N11;
				 //float N22;

				 //float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				 //N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				 //N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				 // N_TM_sp replaced by 	new_TM_sp to converse registers
				 //new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
				 new_TM_sp = fmaf(model_constants[14], new_speed_ind,/*ind*/new_TM_sp);
				 new_TM_sp = new_TM_sp*R_tm_ind;
				 new_TM_sp = new_TM_sp*aux_TM;
				 new_TM_sp = fmaf(S_tm_ind, deltaL_disp, new_TM_sp);
				 //new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
				 // P_TM_vec temporary used here to conserve registers, sorry for the mess
				 // P_TM_vec[threadIdx.x] -> P_tm_ind 
				 //P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 P_tm_ind = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_tm_ind);
				 // Denominator:
				 //D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				 //float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				 // D_TM_Sp will replaced by aux_TM to conserve registers...
				 aux_TM = gamma_ind*aux_TM;
				 aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
				 aux_TM = R_tm_ind*aux_TM;
				 new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				 // Calc_Ptm
				 //_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				 P_tm_ind = S_tm_ind*new_TM_disp;
				 P_tm_ind = fmaf(R_tm_ind, new_TM_sp, P_tm_ind);



			 }
			 // Calc_G   
			 //_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			 // calc R_bm_nl

			 //float R_bm_nl;  replaced by G_ind to converse registers
			 //if (alpha_r == 0.0)
			 //	R_bm_nl = R_ind;
			 //else
			 //float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // G_ind calculation deconstructed from 
			 // (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // float G_ind = new_speed_ind * new_speed_ind;
			 //G_ind = fmaf(model_constants[25], G_ind, 1.0f);
			 //G_ind = R_ind*G_ind;
			 //float G_ind = R_ind;
			 //G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
			 float G_ind = fmaf(R_ind, new_speed_ind/*ind*/, S_bm_ind * new_disp_ind/*ind*/);
			 //G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, Sd_const[wrap_id] * new_disp_vec[threadIdx.x]/*ind*/);
			 //G_ind = fmaf(G_ind, -1.0f, -P_tm_ind);
			 G_ind = -G_ind - P_tm_ind;
			 //G_ind = -1.0f*G_ind;
			 //float dx_pow2 = delta_x * delta_x;


			 float Y_ind;

			 // Calc BC	& Calc_Y   		
			 // _Y						= dx_pow2 * _G * _model._Q;	 
			 // _Y[0]					= _bc;	 
			 // _Y[_model._sections-1]	= 0.0;

			 //float new_sample; // will be sx[8]
			 //float _bc;

			 if (threadIdx.x == 0) {
				 //long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				 // relative position of x in the interval
				 //float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				 //float delta = __fdividef(curr_time_f, model_constants[0]);
				 //float delta = curr_time_f* model_constants[1];
				 sx[9] = curr_time_f* model_constants[1];
				 sx[10] = 1.0f - sx[9];
				 sx[8] = fmaf(sx[1], sx[9], sx[0] * sx[10]);



				 /*
				 if (enable_OW)
				 {
				 _bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				 }
				 else
				 {
				 _bc = delta_x * model_a0 * model_Gme *new_sample;
				 }
				 now I can avoid control
				 however since enable_OW = model_constants_integers[1] is always one lets short the formula
				 Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				 */
				 /**
				 * deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				 */
				 Y_ind = model_constants[32] * sx[6];
				 Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				 Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				 Y_ind = model_constants[26] * Y_ind;
				 // also model_constants[17] * sx[8] is removed since enable_OW is always one
				 //Y_ind = _bc;
			 }
			 else if (threadIdx.x == (blockDim.x - 1)) {
				 Y_ind = 0.0f;
			 }
			 else {
				 Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			 }
			 __syncthreads();
			 // float _bc = new_sample; 



			 // Jacobby -> Y
			 int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			 for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				 __syncthreads();
				 float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 2]);
				 tmp_p = Y_ind - tmp_p;
				 // u_vec  is all 1's so it's removed
				 //l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				 tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				 __syncthreads();




				 pressure_vec[threadIdx.x + 1] = tmp_p;


				 // __threadfence();
			 }

			 __syncthreads();

			 // Calc_BM_acceleration	  
			 //_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			 //new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			 new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			 new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			 //__syncthreads(); 

			 // assuming model_constants_integers[1] && enable_OW always active
			 if (threadIdx.x == 0) {

				 //Calc_OW_Acceleration
				 //_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				 //		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				 sx[9] = model_constants[31] * sx[4];
				 sx[9] = fmaf(model_constants[32], sx[6], sx[9]);
				 sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
				 sx[10] = sx[9] + sx[10];
				 sx[2] = model_constants[7] * sx[10];


			 }


			 // Calc_Ohc_Psi_deriv
			 //_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			 // __syncthreads();
			 /**
			 deconstructing
			 new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			 */
			 new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			 new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			 new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
			 new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
			 __syncthreads();

			 //if (threadIdx.x<3)
			 //{
			 //	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			 //} 


			 //////////////////////////// TESTER //////////////////////////////



			 //float prev_time_step = curr_time_step_f;

			 // Lipschitz condition number
			 //float	Lipschitz;			

			 // find the speed error (compared to the past value)
			 // rempved Lipschits_en due to unecessacity	
			 if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			 {

				 acc_sp_fault_ind.x = fabs(new_accel_ref - new_accel_ind);
				 //acc_diff_ind = blockReduceSum<float, fmaxf >(acc_diff_ind);
				 //sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 acc_sp_fault_ind.y = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 //sp_err_ind = blockReduceSum<float, addition<float>  >(sp_err_ind);
				 //float sp_err_limit = ;
				 // deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				 // fault_vec => fault_ind

				 //fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				 acc_sp_fault_ind.z = acc_sp_fault_ind.y - fmax(mtt, new_speed_ind * curr_time_step_f);
				 //fault_ind = blockReduceSum<float, addition<float>  >(fault_ind);
				 acc_sp_fault_ind = blockReduceTripleAggregators<fmaxf, addition<float>, addition<float> >(acc_sp_fault_ind);
				 // TODO - take only sample into account?
				 __syncthreads();


				 // calculate lipschitz number 
				 if (threadIdx.x<32) {
					 float m1_sp_err = acc_sp_fault_ind.y;	 // shared on single warp not necessary
					 float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									   // Lipschitz calculations (threads 0-32)
					 if (m1_sp_err > mmspt) {
						 Lipschitz = acc_sp_fault_ind.x * curr_time_step_f;
						 Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					 }
					 else
						 Lipschitz = 0.0f;

					 //float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					 //int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					 // use the calculated values to decide


					 if (Lipschitz > 6.0f) //2.0 )
					 {
						 // iteration didn't pass, step size decreased
						 curr_time_step_f *= 0.5f;
						 if (curr_time_step_f<model_constants[21]) {
							 curr_time_step_f = model_constants[21];
						 }

						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul >2
					 else if (acc_sp_fault_ind.z > 0.0f) {
						 // another iteration is needed for this step 
						 another_loop = 1;
						 reset_iteration = 0;
					 } // faults > 0
					 else if (Lipschitz < 0.5f)//(float)(0.25) )
					 {
						 // iteration passed, step size increased 
						 curr_time_step_f *= 2.0f;
						 if (curr_time_step_f>model_constants[22]) {
							 curr_time_step_f = model_constants[22];
						 }
						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul <0.25
					 else {
						 another_loop = 1;
						 reset_iteration = 0;
					 }
					 sx[18] = curr_time_step_f; // broadcast result
					 sx[19] = (float)another_loop;
					 sx[20] = (float)reset_iteration;

				 } // threadIdx.x < 32





			 } // !first_iteration
			 else {

				 if (threadIdx.x < 32) {
					 sx[18] = curr_time_step_f;
					 sx[19] = 1.0f; // another loop 
					 sx[20] = 0.0f; // reset_iteration 
				 }

				 //another_loop = 1;	 
				 //reset_iteration = 0;
			 }


			 __syncthreads();


			 curr_time_step_f = sx[18];
			 another_loop = rint(sx[19]);
			 reset_iteration = rint(sx[20]);

			 /////////////////////////////////////////////////////////



			 out_loop++;
			 // oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			 // out_loop >= _CUDA_OUTERN_LOOPS
			 // convergence has failed
			 if (out_loop >= model_constants_integers[11]) {
				 another_loop = 0;
				 reset_iteration = 1;
				 sx[11] = 1;
			 }
			 //if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		 } // end of outern loop
		 another_loop = 1;
		 //float tdiff = curr_time_f - common_args_vec[0];
		 //tdiff = tdiff > 0 ? tdiff : -tdiff;
		// if (threadIdx.x == 0) {
			 // sx[13] += 1.0f; // counter of iteration;
			 // sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			 // sx[15] += sx[14];
		// }
		 // if overlap time passed or its transient write and I'm actually write it	
		 // replaced curr_time_f >= common_args_vec[0]
		 if (curr_time_f > model_constants[20]) {
			 // TODO
			 curr_time_f -= model_constants[20];
			 if (threadIdx.x == 0) {
				 sx[0] = sx[1];
				 sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			 }
			 __syncthreads();
			 // out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			 if (input_sample + 1 >= mosi) {

				 //int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				 // index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				 // out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				 //int t_ind = ;


				 //saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				 saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				 if (threadIdx.x == 0) {
					 Failed_Converged_Time_Node[input_sample] = sx[11];
					 if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					 // float div_times = __fdividef(sx[12], sx[13]);
					 // convergence_time_measurement[input_sample] = div_times;
					 // convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					 // float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					 // convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					 // convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					 // convergence_delta_time_iterations[input_sample] = sx[13];
					 // convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				 }
				 //__syncthreads();


				 //if (threadIdx.x == 0) {
				 //	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				 //}


				 //out_sample++;


			 }
			 else {
				 saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			 }
			 if (threadIdx.x == 0) {
				 sx[11] = 0;
				 sx[12] = 0;
				 sx[13] = 0;
				 sx[15] = 0;
			 }
			 input_sample++;
			 //__syncthreads(); // NEW NEW 
			 //if (threadIdx.x == 0) {
			 //}
		 } // if write data is needed




		   // copy new to past



		 prev_disp_ind = new_disp_ind; //ind;
		 prev_speed_ind = new_speed_ind/*ind*/;
		 prev_accel_ind = new_accel_ind;
		 prev_ohc_psi_ind = new_ohc_psi_ind;
		 prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		 prev_TM_sp_ind = new_TM_sp;
		 prev_TM_disp_ind = new_TM_disp;

		 // model_constants_integers[1] && enable_OW is always 1
		 if (threadIdx.x == 0) {
			 sx[3] = sx[4];
			 sx[5] = sx[6];
			 sx[7] = sx[2];
		 }

		 __syncthreads();
		 curr_time_f += curr_time_step_f;




	 } // end of sample loop

	   // store data in global mem

	   // TODO - signal finished section
	   // wait for new section to be ready (jumps of ...)



	 __syncthreads();
 }

 // BM calculation kernel, while less finely divided on cochlea property consume less memory
 __global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, KERNEL_BLOCKS) BMOHC_IMPERCISE_kernel(
	 float * __restrict__ input_samples,
	 volatile float *saved_speeds,
	 int * __restrict__ Failed_Converged_Time_Node,
	 int * __restrict__ Failed_Converged_Blocks,
	 //volatile float *saved_speeds_buffer,
	 volatile float *mass,
	 volatile float *rsM, // reciprocal mass for multiplication instead of division
	 volatile float *U,
	 volatile float *L,

	 volatile float *R,
	 volatile float *S,
	 volatile float *Q,

	 volatile float *gamma,
	 volatile float *S_ohc,
	 volatile float *S_tm,
	 volatile float *R_tm,
	 float * __restrict__ gen_model_throw_tolerance,
	 float * __restrict__ gen_model_max_m1_sp_tolerance,
	 int * __restrict__ gen_model_out_sample_index,
	 int * __restrict__ gen_model_end_sample_index,
	 float * __restrict__ convergence_time_measurement,
	 float * __restrict__ convergence_time_measurement_blocks,
	 float * __restrict__ convergence_delta_time_iterations,
	 float * __restrict__ convergence_delta_time_iterations_blocks,
	 float * __restrict__ convergence_jacoby_loops_per_iteration,
	 float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
 ) {

	 int mosi = gen_model_out_sample_index[blockIdx.x];
	 int mesi = gen_model_end_sample_index[blockIdx.x];
	 float mtt = gen_model_throw_tolerance[blockIdx.x];
	 float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];

	 //__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	 //__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	 //__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	 float rsm_ind;
	 //float l_value;
	 //float u_value;
	 __shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	 //__shared__ float curr_time[WARP_SIZE];
	 //__shared__ float curr_time_step[WARP_SIZE];
	 //__shared__ float half_curr_time_step[WARP_SIZE];
	 __shared__ float sx[SX_SIZE];
	 //__shared__ int common_ints[COMMON_INTS_SIZE];
	 //__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	 __shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	 __shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	 //__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	 float P_tm_ind;
	 //__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	 float new_disp_ind;
	 //__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	 float new_speed_ind = 0.0f;
	 //float new_speed_value;
	 // copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	 //__shared__ float common_args_vec[1]; 
	 float curr_time_step_f;
	 float half_curr_time_step_f;
	 float curr_time_f;
	 //float float_base_time_f;  
	 //float next_block_first_sample_f; 
	 float S_bm_ind;
	 float new_ohcp_deriv_ind;
	 float new_ohc_psi_ind;
	 float new_TM_disp;
	 float new_TM_sp;

	 float prev_disp_ind;
	 float prev_speed_ind;
	 float prev_accel_ind;
	 float prev_ohc_psi_ind;
	 float prev_ohcp_deriv_ind;
	 //float prev_OW_displacement_ind;//sx[3]
	 //float prev_OW_acceleration_ind; // sx[7]
	 //float prev_OW_speed_ind;	// sx[5]
	 //float new_OW_displacement_ind; // sx[4]
	 //float new_OW_acceleration_ind;   //sx[2]
	 //float new_OW_speed_ind; // sx[6]
	 float prev_TM_disp_ind;
	 float prev_TM_sp_ind;




	 float new_ohc_pressure_ind;



	 float gamma_ind;

	 float R_ind;

	 //float new_disp_ind;
	 //float new_speed_ind;
	 float new_accel_ind;



	 float Q_ind;
	 //float mass_ind;
	 float reciprocal_mass_ind; // for faster calculations
	 float S_ohc_ind;
	 float S_tm_ind;
	 float R_tm_ind;

	 //int wrap_id = threadIdx.x >> 5;


	 prev_TM_disp_ind = 0.0f;
	 new_TM_disp = 0.0f;
	 prev_TM_sp_ind = 0.0f;
	 new_TM_sp = 0.0f;

	 //prev_OW_displacement_ind = 0.0;	  //sx[3]
	 //new_OW_displacement_ind = 0;			//sx[4]
	 //prev_OW_acceleration_ind = 0;
	 //new_OW_acceleration_ind = 0;
	 //prev_OW_speed_ind = 0;
	 //new_OW_speed_ind = 0;   

	 prev_disp_ind = 0.0f;
	 prev_speed_ind = 0.0f;
	 prev_accel_ind = 0.0f;

	 new_disp_ind = 0.0f;
	 //new_disp_vec[threadIdx.x] = 0.0f;
	 //new_speed_ind = 0; 
	 //new_speed_vec[threadIdx.x] = 0; 
	 //new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	 new_accel_ind = 0.0f;

	 //rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	 rsm_ind = rsM[threadIdx.x];
	 //u_vec[threadIdx.x] = U[threadIdx.x];
	 //l_vec[threadIdx.x] = L[threadIdx.x];
	 //u_value = U[threadIdx.x];
	 //l_value = L[threadIdx.x];
	 pressure_vec[threadIdx.x + 1] = 0.0f;
	 /*
	 if (threadIdx.x==0)
	 {
	 pressure_vec[0] = 0.0;
	 pressure_vec[SECTIONS+1] = 0.0;
	 m_vec[0] = 0.0;
	 m_vec[SECTIONS + 1] = 0.0;
	 u_vec[0] = 0.0;
	 u_vec[SECTIONS + 1] = 0.0;
	 l_vec[0] = 0.0;
	 l_vec[SECTIONS + 1] = 0.0;
	 }
	 */

	 if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	 if (threadIdx.x == 0) {
		 sx[11] = 0;
		 pressure_vec[0] = 0.0;
		 pressure_vec[SECTIONS + 1] = 0.0f;
		 Failed_Converged_Blocks[blockIdx.x] = 0;
		 convergence_time_measurement_blocks[blockIdx.x] = 0;
		 convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		 convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
		 sx[15] = 0;
	 }
	 __syncthreads();
	 prev_ohc_psi_ind = 0.0f;
	 prev_ohcp_deriv_ind = 0.0f;
	 new_ohc_psi_ind = 0.0f;



	 //mass_ind = mass[threadIdx.x]; 
	 reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	 gamma_ind = gamma[threadIdx.x];
	 R_ind = R[threadIdx.x];
	 //S_vec[threadIdx.x] = S[threadIdx.x];
	 S_bm_ind = S[threadIdx.x];
	 Q_ind = Q[threadIdx.x];
	 S_ohc_ind = S_ohc[threadIdx.x];
	 S_tm_ind = S_tm[threadIdx.x];
	 R_tm_ind = R_tm[threadIdx.x];


	 curr_time_step_f = model_constants[19];	   //time_step
	 half_curr_time_step_f = 0.5f*curr_time_step_f;
	 // time offset calculated by nodes and transfered for  float now
	 //int time_offset = nodes_per_time_block*blockIdx.x;
	 //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	
	 //float_base_time_f = 0.0f;
	 //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


	 //if (threadIdx.x == 0) {
	 //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
	 //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
	 // isDecoupled = blockIdx.x%isDecoupled; 
	 // isDecoupled = isDecoupled > 0 ? 0 : 1;
	 // int preDecoupled
	 //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

	 //}
	 // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
	 // Main Algorithm
	 // will be 1 if the next block will be decoupled from this block
	 //int preDecoupled = Decouple_Filter;
	 // will be one if this block is decoupled from last block
	 //int isDecoupled = Decouple_Filter;



	 int start_input_index = ((model_constants_integers[0] > 1 ? (blockIdx.x - (blockIdx.x%model_constants_integers[0])) : 0) * model_constants_integers[8]);

	 int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
	 curr_time_f = float(input_sample - start_input_index)*model_constants[0]; // current time changed to start from specific interval input_sample*model_constants[0];// float(time_offset*time_step_out); // cahnged to be base on time step
	 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
	 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
	 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
	 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
	 //first time to output -> starting after first overlaping period
	 //if (threadIdx.x == 0) {
	 // in case of transient write, start write from the begginning
	 // 
	 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
	 //}


	 //first output time of next block
	 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
	 // removed time_offset +
	 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
	 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
	 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
	 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
	 */
	 //offset for first output sample (in units of samples in output array)
	 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
	 //
	 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	 int another_loop = 1;


	 //int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	 //int Lipschits_en = 1; // unecessary removed



	 //float sx1 = input_samples[input_sample];
	 //float sx2 = input_samples[input_sample+1];
	 if (threadIdx.x <2) {
		 sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	 }
	 __syncthreads();
	 //float P_TM;
	 //float m_inv = 1.0/M[threadIdx.x];

	 // curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	 // previous test bound curr_time_f<next_block_first_sample_f
	 // (out_sample<nodes_per_time_block)
	 // curr_time_f<next_block_first_sample_f
	 while (input_sample<mesi) {



		 __syncthreads();

		 // if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		 // First Step - make approximation using EULER/MEULER


		 for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			 float new_speed_ref = new_speed_ind; //ind;
			 float new_accel_ref = new_accel_ind;

			 if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			 {
				 out_loop = 0;

				 new_TM_disp = fmaf(prev_TM_sp_ind, curr_time_step_f, prev_TM_disp_ind);

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 //new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				 new_speed_ind/*ind*/ = fmaf(prev_accel_ind, curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, half_curr_time_step_f, prev_disp_ind);
				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = fmaf(sx[5], curr_time_step_f, sx[3]);
					 sx[6] = fmaf(sx[7], curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 // OHC:  
				 new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind, curr_time_step_f, prev_ohc_psi_ind);



			 } else		//  TRAPEZOIDAL 
			 {

				 // BM displacement & speed  
				 //new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				 new_speed_ind = prev_accel_ind + new_accel_ind;
				 new_speed_ind/*ind*/ = fmaf(new_speed_ind, half_curr_time_step_f, prev_speed_ind);

				 new_disp_ind = prev_speed_ind + new_speed_ind;
				 new_disp_ind = fmaf(new_disp_ind, half_curr_time_step_f, prev_disp_ind);
				 // not enough shared mem for trapezoidal 
				 // new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				 // model_constants_integers[1] && - assuming enbable_OW always active
				 if (threadIdx.x == 0) {
					 sx[4] = sx[5] + sx[6];
					 sx[4] = fmaf(sx[4], half_curr_time_step_f, sx[3]);
					 sx[6] = sx[7] + sx[2];
					 sx[6] = fmaf(sx[6], half_curr_time_step_f, sx[5]);
				 }
				 __syncthreads();
				 new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				 new_TM_disp = fmaf(new_TM_disp, half_curr_time_step_f, prev_TM_disp_ind);

				 // OHC: 
				 new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				 new_ohc_psi_ind = fmaf(new_ohc_psi_ind, half_curr_time_step_f, prev_ohc_psi_ind);

			 }




			 //_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			 float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			 new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			 // Calc_DeltaL_OHC:
			 // 
			 // if (true == _model._OHC_NL_flag)
			 //	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			 // else
			 //	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			 if (out_loop<CUDA_AUX_TM_LOOPS) {
				 float deltaL_disp;
				 float aux_TM;
				 //if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				 //{
				 //float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				 //float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
				 aux_TM = model_constants[6] * new_ohc_psi_ind;
				 aux_TM = tanhf(aux_TM);
				 deltaL_disp = model_constants[5] * aux_TM;

				 //Calc_TM_speed
				 //aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));

				 //}
				 //else
				 //{
				 //	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				 //Calc_TM_speed
				 //aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				 //	aux_TM = model_constants[6] * new_ohc_psi_ind;
				 //   aux_TM = aux_TM*aux_TM;

				 //}

				 //Calc_TM_speed	 	
				 // aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				 //aux_TM = model_constants[2] * (aux_TM - 1.0);
				 aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
				 aux_TM = model_constants[2] * aux_TM;


				 // Numerator:
				 //N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				 //			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				 //			+ _model._S_tm*_deltaL_disp );

				 //float N11;
				 //float N22;

				 //float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				 //N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				 //N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				 // N_TM_sp replaced by 	new_TM_sp to converse registers
				 //new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = model_constants[18] * new_ohc_psi_ind;
				 new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
				 new_TM_sp = fmaf(model_constants[14], new_speed_ind,/*ind*/new_TM_sp);
				 new_TM_sp = new_TM_sp*R_tm_ind;
				 new_TM_sp = new_TM_sp*aux_TM;
				 new_TM_sp = fmaf(S_tm_ind, deltaL_disp, new_TM_sp);
				 //new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
				 // P_TM_vec temporary used here to conserve registers, sorry for the mess
				 // P_TM_vec[threadIdx.x] -> P_tm_ind 
				 //P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 P_tm_ind = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				 new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_tm_ind);
				 // Denominator:
				 //D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				 //float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				 // D_TM_Sp will replaced by aux_TM to conserve registers...
				 aux_TM = gamma_ind*aux_TM;
				 aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
				 aux_TM = R_tm_ind*aux_TM;
				 new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				 // Calc_Ptm
				 //_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				 P_tm_ind = S_tm_ind*new_TM_disp;
				 P_tm_ind = fmaf(R_tm_ind, new_TM_sp, P_tm_ind);



			 }
			 // Calc_G   
			 //_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			 // calc R_bm_nl

			 //float R_bm_nl;  replaced by G_ind to converse registers
			 //if (alpha_r == 0.0)
			 //	R_bm_nl = R_ind;
			 //else
			 //float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 // G_ind calculation deconstructed from 
			 // (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			 float G_ind = new_speed_ind * new_speed_ind;
			 G_ind = fmaf(model_constants[25], G_ind, 1.0f);
			 G_ind = R_ind*G_ind;

			 //G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
			 G_ind = fmaf(G_ind, new_speed_ind/*ind*/, S_bm_ind * new_disp_ind/*ind*/);
			 //G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, Sd_const[wrap_id] * new_disp_vec[threadIdx.x]/*ind*/);
			 //G_ind = fmaf(G_ind, -1.0f, -P_tm_ind);
			 G_ind = G_ind + P_tm_ind;
			 G_ind = -1.0f*G_ind;
			 //float dx_pow2 = delta_x * delta_x;


			 float Y_ind;

			 // Calc BC	& Calc_Y   		
			 // _Y						= dx_pow2 * _G * _model._Q;	 
			 // _Y[0]					= _bc;	 
			 // _Y[_model._sections-1]	= 0.0;

			 //float new_sample; // will be sx[8]
			 //float _bc;

			 if (threadIdx.x == 0) {
				 //long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				 // relative position of x in the interval
				 //float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				 //float delta = __fdividef(curr_time_f, model_constants[0]);
				 //float delta = curr_time_f* model_constants[1];
				 float base_calculated_time = curr_time_f* model_constants[1];
				 sx[9] = base_calculated_time - floor(base_calculated_time);
				 sx[10] = 1.0f - sx[9];
				 sx[8] = fmaf(sx[1], sx[9], sx[0] * sx[10]);



				 /*
				 if (enable_OW)
				 {
				 _bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				 }
				 else
				 {
				 _bc = delta_x * model_a0 * model_Gme *new_sample;
				 }
				 now I can avoid control
				 however since enable_OW = model_constants_integers[1] is always one lets short the formula
				 Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				 */
				 /**
				 * deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				 */
				 Y_ind = model_constants[32] * sx[6];
				 Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				 Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				 Y_ind = model_constants[26] * Y_ind;
				 // also model_constants[17] * sx[8] is removed since enable_OW is always one
				 //Y_ind = _bc;
			 } else if (threadIdx.x == (blockDim.x - 1)) {
				 Y_ind = 0.0f;
			 } else {
				 Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			 }
			 __syncthreads();
			 // float _bc = new_sample; 



			 // Jacobby -> Y
			 int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			 for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				 __syncthreads();
				 float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 2]);
				 tmp_p = Y_ind - tmp_p;
				 // u_vec  is all 1's so it's removed
				 //l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				 tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				 __syncthreads();




				 pressure_vec[threadIdx.x + 1] = tmp_p;


				 // __threadfence();
			 }

			 __syncthreads();

			 // Calc_BM_acceleration	  
			 //_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			 //new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			 new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			 new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			 //__syncthreads(); 

			 // assuming model_constants_integers[1] && enable_OW always active
			 if (threadIdx.x == 0) {

				 //Calc_OW_Acceleration
				 //_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				 //		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				 sx[9] = model_constants[31] * sx[4];
				 sx[9] = fmaf(model_constants[32], sx[6], sx[9]);
				 sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
				 sx[10] = sx[9] + sx[10];
				 sx[2] = model_constants[7] * sx[10];


			 }


			 // Calc_Ohc_Psi_deriv
			 //_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			 // __syncthreads();
			 /**
			 deconstructing
			 new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			 */
			 new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			 new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			 new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
			 new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
			 __syncthreads();

			 //if (threadIdx.x<3)
			 //{
			 //	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			 //} 


			 //////////////////////////// TESTER //////////////////////////////



			 //float prev_time_step = curr_time_step_f;

			 // Lipschitz condition number
			 //float	Lipschitz;			

			 // find the speed error (compared to the past value)
			 // rempved Lipschits_en due to unecessacity	
			 if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			 {

				 //float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				 acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				 sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				 //float sp_err_limit = ;
				 // deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				 // fault_vec => fault_ind
				 float fault_ind = new_speed_ind * curr_time_step_f;
				 fault_ind = fmax(mtt, fault_ind);
				 fault_ind = sp_err_vec[threadIdx.x] - fault_ind;
				 // TODO - take only sample into account?
				 for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					 __syncthreads();
					 if (threadIdx.x<t_i) {
						 sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						 //fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						 acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					 }

				 }

				 int fault_int = __syncthreads_or(fault_ind > 0.0f);


				 // calculate lipschitz number 
				 if (threadIdx.x<32) {
					 float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					 float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
					 // Lipschitz calculations (threads 0-32)
					 if (m1_sp_err > mmspt) {
						 Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						 Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					 } else
						 Lipschitz = 0.0f;

					 //float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					 //int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					 // use the calculated values to decide


					 if (Lipschitz > 6.0f) //2.0 )
					 {
						 // iteration didn't pass, step size decreased
						 curr_time_step_f *= 0.5f;
						 if (curr_time_step_f<model_constants[21]) {
							 curr_time_step_f = model_constants[21];
						 }

						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul >2
					 else if (fault_int > 0) {
						 // another iteration is needed for this step 
						 another_loop = 1;
						 reset_iteration = 0;
					 } // faults > 0
					 else if (Lipschitz < 0.5f)//(float)(0.25) )
					 {
						 // iteration passed, step size increased 
						 curr_time_step_f *= 2.0f;
						 if (curr_time_step_f>model_constants[22]) {
							 curr_time_step_f = model_constants[22];
						 }
						 another_loop = 0;
						 reset_iteration = 1;
					 } // Lmul <0.25
					 else {
						 another_loop = 1;
						 reset_iteration = 0;
					 }
					 sp_err_vec[0] = curr_time_step_f; // broadcast result
					 sp_err_vec[1] = (float)another_loop;
					 sp_err_vec[2] = (float)reset_iteration;

				 } // threadIdx.x < 32





			 } // !first_iteration
			 else {

				 if (threadIdx.x < 32) {
					 sp_err_vec[0] = curr_time_step_f;
					 sp_err_vec[1] = 1.0f; // another loop 
					 sp_err_vec[2] = 0.0f; // reset_iteration 
				 }

				 //another_loop = 1;	 
				 //reset_iteration = 0;
			 }


			 __syncthreads();


			 curr_time_step_f = sp_err_vec[0];
			 another_loop = rint(sp_err_vec[1]);
			 reset_iteration = rint(sp_err_vec[2]);

			 half_curr_time_step_f = 0.5f*curr_time_step_f;

			 /////////////////////////////////////////////////////////



			 out_loop++;
			 // oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			 // out_loop >= _CUDA_OUTERN_LOOPS
			 if (out_loop >= model_constants_integers[11]) {
				 another_loop = 0;
				 reset_iteration = 1;
				 sx[11] = 1;
			 }
			 if (threadIdx.x == 0) sx[14] = float(out_loop);
		 } // end of outern loop
		 another_loop = 1;
		 //float tdiff = curr_time_f - common_args_vec[0];
		 //tdiff = tdiff > 0 ? tdiff : -tdiff;
		 if (threadIdx.x == 0) {
			 sx[13] += 1.0f; // counter of iteration;
			 sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			 sx[15] += sx[14];
		 }
		 // if overlap time passed or its transient write and I'm actually write it	
		 // replaced curr_time_f >= common_args_vec[0]
		 if (curr_time_f > (input_sample+1 - start_input_index)*model_constants[20]) {
			 // TODO
			 //curr_time_f -= model_constants[20];
			 if (threadIdx.x == 0) {
				 sx[0] = sx[1];
				 sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			 }
			 __syncthreads();
			 // out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			 if (input_sample + 1 >= mosi) {

				 //int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				 // index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				 // out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				 //int t_ind = ;


				 //saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				 saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				 //__syncthreads();
				 if (threadIdx.x == 0) {
					 Failed_Converged_Time_Node[input_sample] = sx[11];
					 if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					 float div_times = __fdividef(sx[12], sx[13]);
					 convergence_time_measurement[input_sample] = div_times;
					 convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					 float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					 convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					 convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					 convergence_delta_time_iterations[input_sample] = sx[13];
					 convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
					 
				 }

				 //if (threadIdx.x == 0) {
				 //	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				 //}


				 //out_sample++;


			 } else {
				 saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			 }
			 if (threadIdx.x == 0) {
				 sx[11] = 0;
				 sx[12] = 0;
				 sx[13] = 0;
				 sx[15] = 0;
			 }
			 input_sample++;
			 //__syncthreads(); // NEW NEW 
			 //if (threadIdx.x == 0) {
			 //}
		 } // if write data is needed




		 // copy new to past



		 prev_disp_ind = new_disp_ind; //ind;
		 prev_speed_ind = new_speed_ind/*ind*/;
		 prev_accel_ind = new_accel_ind;
		 prev_ohc_psi_ind = new_ohc_psi_ind;
		 prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		 prev_TM_sp_ind = new_TM_sp;
		 prev_TM_disp_ind = new_TM_disp;

		 // model_constants_integers[1] && enable_OW is always 1
		 if (threadIdx.x == 0) {
			 sx[3] = sx[4];
			 sx[5] = sx[6];
			 sx[7] = sx[2];
		 }

		 __syncthreads();
		 curr_time_f += curr_time_step_f;




	 } // end of sample loop

	 // store data in global mem

	 // TODO - signal finished section
	 // wait for new section to be ready (jumps of ...)



	 __syncthreads();
 }

__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 2) BMOHC_NEW_kernel(
float * __restrict__ input_samples,
volatile float *saved_speeds,
int * __restrict__ Failed_Converged_Time_Node,
int * __restrict__ Failed_Converged_Blocks,
volatile float *mass,
volatile float *rsM, // reciprocal mass for multiplication instead of division
volatile float *U,
volatile float *L,

volatile float *R,
volatile float *S,
volatile float *Q, 
 
volatile float *gamma,
volatile float *S_ohc,
volatile float *S_tm,
volatile float *R_tm,
float * __restrict__ gen_model_throw_tolerance,
float * __restrict__ gen_model_max_m1_sp_tolerance,
int * __restrict__ gen_model_out_sample_index,
int * __restrict__ gen_model_end_sample_index,
float * __restrict__ convergence_time_measurement,
float * __restrict__ convergence_time_measurement_blocks,
float * __restrict__ convergence_delta_time_iterations,
float * __restrict__ convergence_delta_time_iterations_blocks,
float * __restrict__ convergence_jacoby_loops_per_iteration,
float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];

	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	//float new_speed_value;
     // copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f; 
	float half_curr_time_step_f;
	float curr_time_f; 
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	 
	float new_ohcp_deriv_ind; 
	float new_ohc_psi_ind;
    float new_TM_disp;
    float new_TM_sp;
	 
	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;
     
       
	
	 
	float new_ohc_pressure_ind; 
	   
			  
     
     float gamma_ind;
      
	float R_ind;
	
	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;
	
	    
	
	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;
     
	 
  
  
	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;
     
    //prev_OW_displacement_ind = 0.0;	  //sx[3]
    //new_OW_displacement_ind = 0;			//sx[4]
    //prev_OW_acceleration_ind = 0;
    //new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   
    
     prev_disp_ind = 0.0f;
     prev_speed_ind = 0.0f;
     prev_accel_ind = 0.0f; 
     
     //new_disp_ind = 0;
     new_disp_vec[threadIdx.x] = 0.0f;
     //new_speed_ind = 0; 
     //new_speed_vec[threadIdx.x] = 0; 
	 new_speed_vec[threadIdx.x] = 0.0f;
     new_accel_ind = 0.0f;
      
	rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	 //u_value = U[threadIdx.x];
	 //l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x+1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	  pressure_vec[0] = 0.0;
	  pressure_vec[SECTIONS+1] = 0.0;
	  m_vec[0] = 0.0;
	  m_vec[SECTIONS + 1] = 0.0;
	  u_vec[0] = 0.0;
	  u_vec[SECTIONS + 1] = 0.0;
	  l_vec[0] = 0.0;
	  l_vec[SECTIONS + 1] = 0.0;
	}
	  */

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[11] = 0;
		Failed_Converged_Blocks[blockIdx.x] = 0;
		convergence_time_measurement_blocks[blockIdx.x] = 0;
		convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
		sx[15] = 0;
	}
	__syncthreads();
	  prev_ohc_psi_ind = 0.0f;
	  prev_ohcp_deriv_ind = 0.0f;
	  new_ohc_psi_ind = 0.0f;  
        
      
       
      //mass_ind = mass[threadIdx.x]; 
	  reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
      gamma_ind = gamma[threadIdx.x]; 
      R_ind = R[threadIdx.x];
      S_vec[threadIdx.x] = S[threadIdx.x]; 
      Q_ind = Q[threadIdx.x];
      S_ohc_ind = S_ohc[threadIdx.x];
      S_tm_ind = S_tm[threadIdx.x];
      R_tm_ind = R_tm[threadIdx.x];
      

	  curr_time_step_f = model_constants[19];	   //time_step
	  half_curr_time_step_f = 0.5f*curr_time_step_f;
	  // time offset calculated by nodes and transfered for  float now
	  //int time_offset = nodes_per_time_block*blockIdx.x;
	  //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	  curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
	  //float_base_time_f = 0.0f;
	  //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


	  //if (threadIdx.x == 0) {
		  //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
		//  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
		  // isDecoupled = blockIdx.x%isDecoupled; 
		  // isDecoupled = isDecoupled > 0 ? 0 : 1;
		  // int preDecoupled
		//  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

	  //}
	// if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
		// Main Algorithm
	 // will be 1 if the next block will be decoupled from this block
	 //int preDecoupled = Decouple_Filter;
		 // will be one if this block is decoupled from last block
	 //int isDecoupled = Decouple_Filter;
	 
		
	 
	 int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
	 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
	 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
	 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
	 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
	//first time to output -> starting after first overlaping period
	//if (threadIdx.x == 0) {
		// in case of transient write, start write from the begginning
		// 
	//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
	//}
	 
	
	//first output time of next block
	// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
	// removed time_offset +
	//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
	/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
	next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
	next_block_first_sample_f = next_block_first_sample_f + time_step_out;
	  */
	//offset for first output sample (in units of samples in output array)
	//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
	//
	//int time_offset = rint((curr_time_f - base_time) / time_step_out);
	 
	
	  
		 
	 
	int another_loop = 1;
	
	 
	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed
	
		
		
	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi)
	{	
		
		 
		 
	  __syncthreads();  
        
       // if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 
           
           
      // First Step - make approximation using EULER/MEULER
      
      
		 for (int out_loop=0, reset_iteration=1 ;  another_loop; )
		 { 
	
		 
			 float new_speed_ref = new_speed_vec[threadIdx.x]; //ind;
			float new_accel_ref = new_accel_ind;
      
			if (  (out_loop==0)||reset_iteration) // first iteration -> EULER
			{				
				out_loop = 0;
				
				new_TM_disp = fmaf(prev_TM_sp_ind,curr_time_step_f,prev_TM_disp_ind);
				
				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_vec[threadIdx.x]/*ind*/ = fmaf(prev_accel_ind,curr_time_step_f,prev_speed_ind);
				 
				new_disp_vec[threadIdx.x] = prev_speed_ind + new_speed_vec[threadIdx.x];
				new_disp_vec[threadIdx.x] = fmaf(new_disp_vec[threadIdx.x], half_curr_time_step_f, prev_disp_ind);
				// model_constants_integers[1] && - assuming enbable_OW always active
				if ( threadIdx.x == 0)
				{ 
					sx[4] = fmaf(sx[5],curr_time_step_f,sx[3]);
					sx[6] = fmaf(sx[7],curr_time_step_f,sx[5]);
				}	
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind,curr_time_step_f,prev_ohc_psi_ind);
			 
						

			}
			else		//  TRAPEZOIDAL 
			{
	 
				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_vec[threadIdx.x] = prev_accel_ind + new_accel_ind;
				new_speed_vec[threadIdx.x]/*ind*/ = fmaf(new_speed_vec[threadIdx.x], half_curr_time_step_f, prev_speed_ind);
			   
				new_disp_vec[threadIdx.x] = prev_speed_ind + new_speed_vec[threadIdx.x];
				new_disp_vec[threadIdx.x] = fmaf(new_disp_vec[threadIdx.x], half_curr_time_step_f, prev_disp_ind);
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );
				
			 
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0)
				{ 
					sx[4] = sx[5] + sx[6];
					sx[4] = fmaf(sx[4], half_curr_time_step_f, sx[3]);
					sx[6] = sx[7] + sx[2];
					sx[6] = fmaf(sx[6], half_curr_time_step_f, sx[5]);
				}
				__syncthreads();
				new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				new_TM_disp = fmaf(new_TM_disp, half_curr_time_step_f, prev_TM_disp_ind);
				
				// OHC: 
				new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				new_ohc_psi_ind = fmaf(new_ohc_psi_ind, half_curr_time_step_f, prev_ohc_psi_ind);
				 
			}
	 
	
	   
	 
	//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
	float ohc_disp = new_TM_disp - new_disp_vec[threadIdx.x]; //new_disp_ind;
	new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;
	
	
	
	   
	// Calc_DeltaL_OHC:
	// 
	// if (true == _model._OHC_NL_flag)
	//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
	// else
	//	_deltaL_disp = -_model._alpha_L * _psi_ohc;
	
	if (out_loop<CUDA_AUX_TM_LOOPS)
	{
		float deltaL_disp;
		float aux_TM;
		//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
		//{
			//float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
			//float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
			aux_TM = model_constants[6] * new_ohc_psi_ind;
			aux_TM = tanhf(aux_TM);
			deltaL_disp = model_constants[5] * aux_TM;
		
		//Calc_TM_speed
		//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
			//aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));
			 
		//}
		//else
		//{
		//	deltaL_disp = model_constants[4] * new_ohc_psi_ind;
			
		   //Calc_TM_speed
		   //aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
		//	aux_TM = model_constants[6] * new_ohc_psi_ind;
		//   aux_TM = aux_TM*aux_TM;
		    
		//}
	
		//Calc_TM_speed	 	
		// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
		//aux_TM = model_constants[2] * (aux_TM - 1.0);
		aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
		aux_TM = model_constants[2] * aux_TM;


	// Numerator:
	//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
	//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
	//			+ _model._S_tm*_deltaL_disp );
	
	//float N11;
	//float N22;
	
	//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
	//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
	//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;
	 
		// N_TM_sp replaced by 	new_TM_sp to converse registers
		//new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
		new_TM_sp = model_constants[18] * new_ohc_psi_ind;
		new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
		new_TM_sp = fmaf(model_constants[14], new_speed_vec[threadIdx.x],/*ind*/new_TM_sp);
		new_TM_sp = new_TM_sp*R_tm_ind;
		new_TM_sp = new_TM_sp*aux_TM;
		new_TM_sp = fmaf(S_tm_ind,deltaL_disp , new_TM_sp);
		//new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
		// P_TM_vec temporary used here to conserve registers, sorry for the mess
		P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
		new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_TM_vec[threadIdx.x]);
	// Denominator:
	//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

	//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
	  // D_TM_Sp will replaced by aux_TM to conserve registers...
		aux_TM = gamma_ind*aux_TM;
		aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
		aux_TM = R_tm_ind*aux_TM;
		new_TM_sp = __fdividef(new_TM_sp, aux_TM);
 
	// Calc_Ptm
	//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
		P_TM_vec[threadIdx.x] = S_tm_ind*new_TM_disp;
		P_TM_vec[threadIdx.x] = fmaf(R_tm_ind, new_TM_sp, P_TM_vec[threadIdx.x]);
		
		
		
	  }
	// Calc_G   
    //_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 
    
    // calc R_bm_nl
    
    //float R_bm_nl;  replaced by G_ind to converse registers
	//if (alpha_r == 0.0)
	//	R_bm_nl = R_ind;
	//else
	//float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
	// G_ind calculation deconstructed from 
	// (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
	float G_ind = new_speed_vec[threadIdx.x] * new_speed_vec[threadIdx.x];
	G_ind = fmaf(model_constants[25], G_ind, 1.0f);
	G_ind = R_ind*G_ind;
  
	//G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
	G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/);
	G_ind = fmaf(G_ind, -1.0f, -P_TM_vec[threadIdx.x]);
	
	//float dx_pow2 = delta_x * delta_x;
		

	float Y_ind;
		
	// Calc BC	& Calc_Y   		
	// _Y						= dx_pow2 * _G * _model._Q;	 
	// _Y[0]					= _bc;	 
	// _Y[_model._sections-1]	= 0.0;
	
			//float new_sample; // will be sx[8]
			//float _bc;
		
			if (threadIdx.x==0)
			{ 
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, model_constants[0]);
				//float delta = curr_time_f* model_constants[1];
				sx[9] = curr_time_f* model_constants[1];
				sx[10] = 1.0f - sx[9];
				sx[8] = fmaf( sx[1]  , sx[9] , sx[0]  * sx[10] );
				  
				
				 
				/*
				if (enable_OW)	
				{
					_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				*/
				/**
				* deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				*/
				Y_ind = model_constants[32] * sx[6];
				Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				Y_ind = model_constants[26] * Y_ind;
				// also model_constants[17] * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x==(blockDim.x-1))
			{ 
				Y_ind = 0.0f;
			}
			else
			{ 
				Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			} 
			__syncthreads();
			// float _bc = new_sample; 
	 
	 
	     
			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
				 for (int iter = 0;(iter<j_iters/*_CUDA_JACOBBY_LOOPS*/);iter++)
				{ 
					 __syncthreads();  
					 float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 2]);
					 tmp_p = Y_ind - tmp_p;
					 // u_vec  is all 1's so it's removed
					 //l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
					 tmp_p = tmp_p*rsm_vec[threadIdx.x]; // /m_vec[threadIdx.x+1]; 	 		 
			 		
 	 
					__syncthreads();
					
					
 
					
					pressure_vec[threadIdx.x+1] = tmp_p; 
	 				 

					// __threadfence();
		  		 } 
	  	    
				 __syncthreads();
	  
	// Calc_BM_acceleration	  
	//_BM_acc = ( _pressure + _G ) / _model._M_bm; 
	
				 //new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
				 new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
				 new_accel_ind = new_accel_ind* reciprocal_mass_ind;
	  	  	
	//__syncthreads(); 
	
   		// assuming model_constants_integers[1] && enable_OW always active
		if ( threadIdx.x == 0){  		
		
		//Calc_OW_Acceleration
		//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
		//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
			sx[9] = model_constants[31] * sx[4];
			sx[9] = fmaf(model_constants[32], sx[6],sx[9]);
			sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
			sx[10]=  sx[9] + sx[10];
			sx[2] = model_constants[7] * sx[10];
		
		
		}
		
		
		// Calc_Ohc_Psi_deriv
	   //_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
		__syncthreads();
		/**
		deconstructing
		new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
		*/	
		new_ohcp_deriv_ind = new_TM_sp - new_speed_vec[threadIdx.x];
		new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
		new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
		new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
		__syncthreads(); 
			
//if (threadIdx.x<3)
//{
//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
//} 


//////////////////////////// TESTER //////////////////////////////

 
 
	//float prev_time_step = curr_time_step_f;
 
	// Lipschitz condition number
	//float	Lipschitz;			

	// find the speed error (compared to the past value)
	// rempved Lipschits_en due to unecessacity	
	if ( reset_iteration==0) // if reset_iteration (first time) - do another loop
	{ 
	 
		//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
	 acc_diff_vec[threadIdx.x] = fabs( new_accel_ref - new_accel_ind );
	 sp_err_vec[threadIdx.x] = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
	 //float sp_err_limit = ;
	 // deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
	 fault_vec[threadIdx.x] =  new_speed_vec[threadIdx.x] * curr_time_step_f;
	 fault_vec[threadIdx.x] = fmax(mtt, fault_vec[threadIdx.x]);
	 fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] > fault_vec[threadIdx.x];
		// TODO - take only sample into account?
		for (int t_i=(SECTIONS>>1) ; t_i>=1; t_i >>= 1)
		{
			__syncthreads();
			if (threadIdx.x<t_i) 
			{
			sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x+t_i];
			fault_vec[threadIdx.x]  = fault_vec[threadIdx.x]  + fault_vec[threadIdx.x+t_i];
			acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x+t_i]);
			}
		
		}
	
		__syncthreads();
		 
	
		// calculate lipschitz number 
		if (threadIdx.x<32)
		{
			//float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
			//float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
			// Lipschitz calculations (threads 0-32)
			if (sp_err_vec[0] > mmspt)
				{
					sp_err_vec[1] = acc_diff_vec[0] * curr_time_step_f;
					sp_err_vec[1] = __fdividef(sp_err_vec[1], sp_err_vec[0]);

	  			 }
			else
				sp_err_vec[1] = 0;

			//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition
		 
			//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;
	
			// use the calculated values to decide
	  
	  	 
			if (sp_err_vec[1] > 6.0f) //2.0 )
			{
				// iteration didn't pass, step size decreased
				curr_time_step_f *= 0.5f;
				if (curr_time_step_f<model_constants[21])
				{
					curr_time_step_f = model_constants[21];
				}
			 
				another_loop = 0;		 
				reset_iteration = 1;
			} // Lmul >2
			else if ( fault_vec[0] > 0.0f)
			{
				// another iteration is needed for this step 
				another_loop = 1;	 
				reset_iteration = 0;
			} // faults > 0
			else if (sp_err_vec[1] < 0.5f)//(float)(0.25) )
			{
				// iteration passed, step size increased 
				curr_time_step_f *= 2.0f;
				if (curr_time_step_f>model_constants[22])
				{
					curr_time_step_f = model_constants[22];
				}
				another_loop = 0;	 
				reset_iteration = 1;
			} // Lmul <0.25
			else 
			{  
				 another_loop = 1;	 
				reset_iteration = 0;
			}
			sp_err_vec[0] = curr_time_step_f; // broadcast result
			sp_err_vec[1] = (float)another_loop; 
			sp_err_vec[2] = (float)reset_iteration; 
     
		} // threadIdx.x < 32
     
     
	  
	} // !first_iteration
	else
	{
	
		if (threadIdx.x < 32)
		{
			sp_err_vec[0] = curr_time_step_f;
			sp_err_vec[1] = 1.0f; // another loop 
			sp_err_vec[2] = 0.0f; // reset_iteration 
		}
	 //another_loop = 1;	 
	// reset_iteration = 0;
	}
	 
	 
	 
	__syncthreads();

	 
	curr_time_step_f = sp_err_vec[0];
	another_loop =  rint(sp_err_vec[1]);	
	reset_iteration = rint(sp_err_vec[2]);	 

	half_curr_time_step_f = 0.5f*curr_time_step_f;
 /////////////////////////////////////////////////////////


	
	out_loop++;
	// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
	// out_loop >= _CUDA_OUTERN_LOOPS
	if (out_loop >= model_constants_integers[11])
	{
		another_loop = 0;
		reset_iteration = 1;
		sx[11] = 1;
	}
	if (threadIdx.x == 0) sx[14] = float(out_loop);
	} // end of outern loop
    another_loop = 1;
	//float tdiff = curr_time_f - common_args_vec[0];
	//tdiff = tdiff > 0 ? tdiff : -tdiff;
	if (threadIdx.x == 0) {
		sx[13] += 1.0f; // counter of iteration;
		sx[12] += curr_time_step_f; // accumulating time steps to calculate average
		sx[15] += sx[14];
	}
	// if overlap time passed or its transient write and I'm actually write it	
	// replaced curr_time_f >= common_args_vec[0]
	if (curr_time_f > model_constants[20])
	{
	 // TODO
		curr_time_f -= model_constants[20];
		input_sample++;
		if (threadIdx.x == 0) {
			sx[0] = sx[1];
			sx[1] = input_samples[input_sample + 1];
		}
		__syncthreads();
		// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
		if (input_sample >= mosi) {

			//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
			// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
			// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
			//int t_ind = ;


			saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_vec[threadIdx.x]/*ind*/;
			if (threadIdx.x == 0) {
				Failed_Converged_Time_Node[input_sample] = sx[11];
				if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
				float div_times = __fdividef(sx[12], sx[13]);
				convergence_time_measurement[input_sample] = div_times;
				convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times , float(mesi - mosi));
				float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
				convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
				convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
				convergence_delta_time_iterations[input_sample] = sx[13];
				convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
			}
			//__syncthreads();


			//if (threadIdx.x == 0) {
			//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
			//}


			//out_sample++;


		}

		if (threadIdx.x == 0) {
			sx[11] = 0;
			sx[12] = 0;
			sx[13] = 0;
			sx[15] = 0;
		}
		//__syncthreads(); // NEW NEW 
		//if (threadIdx.x == 0) {
		//}
	} // if write data is needed
 

 

	// copy new to past
	
	     
	     
	prev_disp_ind = new_disp_vec[threadIdx.x]; //ind;
	prev_speed_ind = new_speed_vec[threadIdx.x]/*ind*/;
	prev_accel_ind = new_accel_ind;
	 prev_ohc_psi_ind = new_ohc_psi_ind;
	 prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
	 prev_TM_sp_ind = new_TM_sp;
	 prev_TM_disp_ind = new_TM_disp;
	 
    // model_constants_integers[1] && enable_OW is always 1
	 if ( threadIdx.x == 0)
	{ 
		sx[3] = sx[4];
		sx[5] = sx[6];
		sx[7] = sx[2];
	}

	__syncthreads();
	curr_time_f += curr_time_step_f;
 
	
 
 
} // end of sample loop
			
 // store data in global mem
 
     // TODO - signal finished section
     // wait for new section to be ready (jumps of ...)
     
       
      
	  __syncthreads();
}



/////////////////////////////////////////////////////////////////
 

///////////////////////////////////////////////////

/*
__global__ void Jaccoby_kernel (

float *P,
float *M,
float *U,
float *L,
float *V,
int jt

){
   
  
	__shared__ float p_vec[T_WIDTH+2]; 
	__shared__ float l_vec[T_WIDTH+2];
	__shared__ float u_vec[T_WIDTH+2];  
	__shared__ float m_vec[T_WIDTH+2]; 
	__shared__ float v_vec[T_WIDTH+2];   
     
      
     
     int tx;
     int txpp;
     int indx;
     int base_idx;
     
     tx = threadIdx.x;
     txpp = tx + 1;
     base_idx = blockDim.x * blockIdx.x;
     indx =  base_idx + tx;
    
      
	p_vec[txpp] = P[indx];
// TODO -  load constants once, split between warps
	m_vec[txpp] = M[indx];
	u_vec[txpp] = U[indx];
	l_vec[txpp] = L[indx];
	v_vec[txpp] = V[indx];
	__syncthreads(); 
	  
	  if ((tx==0)&&(indx!=0))
	  {
	    p_vec[0] = P[indx-1];
	    l_vec[0] = L[indx-1];
	  }
	  if ((blockIdx.x!=LAST_BLOCK)&&(tx==(T_WIDTH-1)))
	  {
	   p_vec[txpp+1] = P[indx+1];
	   u_vec[txpp+1] = U[indx+1];
	  }
  
	__syncthreads();
	
	//int edges = (((blockIdx.x==LAST_BLOCK)&&(tx==(T_WIDTH-1)))||(indx==0));
	 	//if (!edges)
	 //	{ 
	     for (int iter = 0;iter<10;iter++)
	     { 
	 		p_vec[txpp] = (v_vec[txpp]-(l_vec[txpp]*p_vec[tx]+u_vec[txpp]*p_vec[txpp+1]))/m_vec[txpp]; 	 		 
	 		
	  	 } 
	   // __syncthreads();
		 
	  // } 
	
 
	// __syncthreads();

	P[indx] = p_vec[txpp]; 
	 
	// __syncthreads();
    
}
*/

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// multiple attempts to roll back changes in order to examine speed of the algorithm
//
/////////////////////////////////////////////////////////////////////////////////////////////////


// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic failures in order to test 47 register per thread run
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 5) BMOHC_FAST_NO_SelfAnalysis_kernel(
	float *input_samples,
	volatile float *saved_speeds,
	//int * __restrict__ Failed_Converged_Time_Node,
	//int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * gen_model_throw_tolerance,
	float * gen_model_max_m1_sp_tolerance,
	int * gen_model_out_sample_index,
	int * gen_model_end_sample_index
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
		// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
		// convergence_time_measurement_blocks[blockIdx.x] = 0;
		// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
		// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	//__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = model_constants[19];	   //time_step
	//half_curr_time_step_f = 0.5f*curr_time_step_f;
	// time offset calculated by nodes and transfered for  float now
	//int time_offset = nodes_per_time_block*blockIdx.x;
	//curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
																 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																 //first time to output -> starting after first overlaping period
																 //if (threadIdx.x == 0) {
																 // in case of transient write, start write from the begginning
																 // 
																 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																 //}


																 //first output time of next block
																 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																 // removed time_offset +
																 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																 */
																 //offset for first output sample (in units of samples in output array)
																 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																 //
																 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = fmaf(prev_TM_sp_ind, curr_time_step_f, prev_TM_disp_ind);

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = fmaf(prev_accel_ind, curr_time_step_f, prev_speed_ind);

				new_disp_ind = prev_speed_ind + new_speed_ind;
				new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = fmaf(sx[5], curr_time_step_f, sx[3]);
					sx[6] = fmaf(sx[7], curr_time_step_f, sx[5]);
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = fmaf(prev_ohcp_deriv_ind, curr_time_step_f, prev_ohc_psi_ind);



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind = prev_accel_ind + new_accel_ind;
				new_speed_ind/*ind*/ = fmaf(new_speed_ind, 0.5f*curr_time_step_f, prev_speed_ind);

				new_disp_ind = prev_speed_ind + new_speed_ind;
				new_disp_ind = fmaf(new_disp_ind, 0.5f*curr_time_step_f, prev_disp_ind);
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] + sx[6];
					sx[4] = fmaf(sx[4], 0.5f*curr_time_step_f, sx[3]);
					sx[6] = sx[7] + sx[2];
					sx[6] = fmaf(sx[6], 0.5f*curr_time_step_f, sx[5]);
				}
				__syncthreads();
				new_TM_disp = prev_TM_sp_ind + new_TM_sp;
				new_TM_disp = fmaf(new_TM_disp, 0.5f*curr_time_step_f, prev_TM_disp_ind);

				// OHC: 
				new_ohc_psi_ind = prev_ohcp_deriv_ind + new_ohcp_deriv_ind;
				new_ohc_psi_ind = fmaf(new_ohc_psi_ind, 0.5f*curr_time_step_f, prev_ohc_psi_ind);

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				//float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//float tan_arg = model_constants[6] * new_ohc_psi_ind; => tan_arg replaced by aux_tm
				aux_TM = model_constants[6] * new_ohc_psi_ind;
				aux_TM = tanhf(aux_TM);
				deltaL_disp = model_constants[5] * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//aux_TM = aux_TM*aux_TM; conserved in line of 	aux_TM = model_constants[2] * (aux_TM - 1.0); => aux_TM = model_constants[2] * (fmaf(aux_TM,aux_TM, -1.0f));

				//}
				//else
				//{
				//	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = model_constants[6] * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = model_constants[2] * (aux_TM - 1.0);
				aux_TM = fmaf(aux_TM, aux_TM, -1.0f);
				aux_TM = model_constants[2] * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				new_TM_sp = model_constants[18] * new_ohc_psi_ind;
				new_TM_sp = fmaf(model_constants[13], ohc_disp, new_TM_sp);
				new_TM_sp = fmaf(model_constants[14], new_speed_ind,/*ind*/new_TM_sp);
				new_TM_sp = new_TM_sp*R_tm_ind*aux_TM;
				new_TM_sp = fmaf(S_tm_ind, deltaL_disp, new_TM_sp);
				//new_TM_sp = fmaf(S_tm_ind,new_TM_disp,new_ohc_pressure_ind) - gamma_ind*new_TM_sp;
				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				//P_TM_vec[threadIdx.x] = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				P_tm_ind = fmaf(S_tm_ind, new_TM_disp, new_ohc_pressure_ind);
				new_TM_sp = fmaf(-gamma_ind, new_TM_sp, P_tm_ind);
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM;
				aux_TM = fmaf(aux_TM, model_constants[14], -1.0f);
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = S_tm_ind*new_TM_disp;
				P_tm_ind = fmaf(R_tm_ind, new_TM_sp, P_tm_ind);



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else
			//float G_ind = (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			// G_ind calculation deconstructed from 
			// (R_ind * fmaf(model_constants[25],new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/,1.0f) );
			//float G_ind = new_speed_ind * new_speed_ind;
			//G_ind = fmaf(model_constants[25], G_ind, 1.0f);
			//G_ind = R_ind*G_ind;

			//G_ind = -1.0f*(P_TM_vec[threadIdx.x] + fmaf(G_ind,new_speed_vec[threadIdx.x]/*ind*/,S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));
			float G_ind = fmaf(R_ind, new_speed_ind/*ind*/, S_bm_ind * new_disp_ind/*ind*/);
			//G_ind = fmaf(G_ind, new_speed_vec[threadIdx.x]/*ind*/, Sd_const[wrap_id] * new_disp_vec[threadIdx.x]/*ind*/);
			G_ind = fmaf(G_ind, -1.0f, -P_tm_ind);
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, model_constants[0]);
				//float delta = curr_time_f* model_constants[1];
				sx[9] = curr_time_f* model_constants[1];
				//sx[10] = 1.0f - sx[9];
				sx[10] = fmaf(sx[9], -1 * sx[0], sx[0]);
				sx[8] = fmaf(sx[1], sx[9],  sx[10]);



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				*/
				/**
				* deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				*/
				Y_ind = model_constants[32] * sx[6];
				Y_ind = fmaf(model_constants[31], sx[4], Y_ind);
				Y_ind = fmaf(model_constants[15], sx[8], Y_ind);
				Y_ind = model_constants[26] * Y_ind;
				// also model_constants[17] * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = fmaf(threadIdx.x<SECTIONSM2, pressure_vec[threadIdx.x], pressure_vec[threadIdx.x + 1 + 1]);
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				sx[9] = model_constants[31] * sx[4];
				sx[9] = fmaf(model_constants[32], sx[6], sx[9]);
				sx[10] = fmaf(model_constants[15], sx[8], pressure_vec[1]);
				sx[10] = sx[9] + sx[10];
				sx[2] = model_constants[7] * sx[10];


			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			/**
			deconstructing
			new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			*/
			new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			new_ohcp_deriv_ind = fmaf(model_constants[13], ohc_disp, new_ohcp_deriv_ind);
			new_ohcp_deriv_ind = fmaf(model_constants[18], new_ohc_psi_ind, new_ohcp_deriv_ind);
			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind
				float fault_ind = new_speed_ind * curr_time_step_f;
				fault_ind = fmax(mtt, fault_ind);
				fault_ind = sp_err_vec[threadIdx.x] - fault_ind;
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						//fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				int fault_int = __syncthreads_or(fault_ind > 0.0f);


				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<model_constants[21]) {
							curr_time_step_f = model_constants[21];
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_int > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>model_constants[22]) {
							curr_time_step_f = model_constants[22];
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				//sx[11] = 1;
			}
			//if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		//if (threadIdx.x == 0) {
			// sx[13] += 1.0f; // counter of iteration;
			// sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			// sx[15] += sx[14];
		//}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > model_constants[20]) {
			// TODO
			curr_time_f -= model_constants[20];
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				//if (threadIdx.x == 0) {
				//	Failed_Converged_Time_Node[input_sample] = sx[11];
				//	if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					// float div_times = __fdividef(sx[12], sx[13]);
					// convergence_time_measurement[input_sample] = div_times;
					// convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					// float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					// convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					// convergence_delta_time_iterations[input_sample] = sx[13];
					// convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				//}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}
			/**
			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}*/
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, KERNEL_BLOCKS) BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_kernel(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	//int * __restrict__ Failed_Converged_Time_Node,
	//int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index
	//float * __restrict__ convergence_time_measurement,
	//float *convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	//__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = model_constants[19];	   //time_step
											   //half_curr_time_step_f = 0.5f*curr_time_step_f;
											   // time offset calculated by nodes and transfered for  float now
											   //int time_offset = nodes_per_time_block*blockIdx.x;
											   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
																 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																 //first time to output -> starting after first overlaping period
																 //if (threadIdx.x == 0) {
																 // in case of transient write, start write from the begginning
																 // 
																 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																 //}


																 //first output time of next block
																 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																 // removed time_offset +
																 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																 */
																 //offset for first output sample (in units of samples in output array)
																 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																 //
																 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f+ prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f+ prev_speed_ind;

				
				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f+ prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5]* curr_time_step_f+ sx[3];
					sx[6] = sx[7]* curr_time_step_f+ sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f+ prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f+ prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f+ prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					
					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f+ sx[3];
					
					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f+ sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f+ prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f+ prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				//float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				float tan_arg = model_constants[6] * new_ohc_psi_ind; //=> tan_arg replaced by aux_tm
				//aux_TM = model_constants[6] * new_ohc_psi_ind;
				aux_TM = tanhf(tan_arg);
				deltaL_disp = model_constants[5] * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				
				//}
				//else
				//{
				//	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = model_constants[6] * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = model_constants[2] * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM -1.0f;
				aux_TM = model_constants[2] * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;
				
				new_TM_sp = model_constants[14]* new_speed_ind + model_constants[13]* ohc_disp+ model_constants[18] * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp+ new_TM_sp*R_tm_ind*aux_TM;
				
				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind -gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* model_constants[14]  -1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp+ S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else
			
			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(model_constants[25]* G_ind+ 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/+ S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind -P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, model_constants[0]);
				//float delta = curr_time_f* model_constants[1];
				sx[9] = curr_time_f* model_constants[1];
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0]*(1.0f-sx[9]);
				sx[8] = sx[1]* sx[9]+ sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				*/
				/**
				* deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				*/
				Y_ind = model_constants[26] * (model_constants[32] * sx[6]+ model_constants[31]* sx[4]+ model_constants[15]* sx[8]);
				// also model_constants[17] * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x]+ pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				sx[9] = model_constants[31] * sx[4];
				sx[9] = model_constants[32]* sx[6]+ sx[9];
				sx[10] = model_constants[15]* sx[8]+ pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = model_constants[7] * sx[10];


			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			/**
			deconstructing
			new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			*/
			new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			new_ohcp_deriv_ind = model_constants[13]* ohc_disp + new_ohcp_deriv_ind;
			new_ohcp_deriv_ind = model_constants[18]* new_ohc_psi_ind + new_ohcp_deriv_ind;
			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind
				float fault_ind = new_speed_ind * curr_time_step_f;
				fault_ind = fmax(mtt, fault_ind);
				fault_ind = sp_err_vec[threadIdx.x] - fault_ind;
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						//fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				int fault_int = __syncthreads_or(fault_ind > 0.0f);


				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<model_constants[21]) {
							curr_time_step_f = model_constants[21];
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_int > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>model_constants[22]) {
							curr_time_step_f = model_constants[22];
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				//sx[11] = 1;
			}
			//if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		//if (threadIdx.x == 0) {
		// sx[13] += 1.0f; // counter of iteration;
		// sx[12] += curr_time_step_f; // accumulating time steps to calculate average
		// sx[15] += sx[14];
		//}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > model_constants[20]) {
			// TODO
			curr_time_f -= model_constants[20];
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
																									//if (threadIdx.x == 0) {
																									//	Failed_Converged_Time_Node[input_sample] = sx[11];
																									//	if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
																									// float div_times = __fdividef(sx[12], sx[13]);
																									// convergence_time_measurement[input_sample] = div_times;
																									// convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
																									// float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
																									// convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
																									// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
																									// convergence_delta_time_iterations[input_sample] = sx[13];
																									// convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
																									//}
																									//__syncthreads();


																									//if (threadIdx.x == 0) {
																									//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
																									//}


																									//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}
			/**
			if (threadIdx.x == 0) {
			sx[11] = 0;
			sx[12] = 0;
			sx[13] = 0;
			sx[15] = 0;
			}*/
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid desgniate numbers as floats
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, KERNEL_BLOCKS) BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Or_Sync_kernel(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	//int * __restrict__ Failed_Converged_Time_Node,
	//int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float *convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0;
	new_TM_disp = 0.0;
	prev_TM_sp_ind = 0.0;
	new_TM_sp = 0.0;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0;
	prev_speed_ind = 0.0;
	prev_accel_ind = 0.0;

	new_disp_ind = 0.0;
	//new_disp_vec[threadIdx.x] = 0.0;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0; => already loaded
	new_accel_ind = 0.0;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0;
	prev_ohcp_deriv_ind = 0.0;
	new_ohc_psi_ind = 0.0;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0 / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = model_constants[19];	   //time_step
											   //half_curr_time_step_f = 0.5*curr_time_step_f;
											   // time offset calculated by nodes and transfered for  float now
											   //int time_offset = nodes_per_time_block*blockIdx.x;
											   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = model_constants_integers[8] * blockIdx.x; // start from the beginng of the block
																 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																 //first time to output -> starting after first overlaping period
																 //if (threadIdx.x == 0) {
																 // in case of transient write, start write from the begginning
																 // 
																 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0;
																 //}


																 //first output time of next block
																 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																 // removed time_offset +
																 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																 */
																 //offset for first output sample (in units of samples in output array)
																 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																 //
																 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				//float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				float tan_arg = model_constants[6] * new_ohc_psi_ind; //=> tan_arg replaced by aux_tm
																	  //aux_TM = model_constants[6] * new_ohc_psi_ind;
				aux_TM = tanhf(tan_arg);
				deltaL_disp = model_constants[5] * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = model_constants[4] * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = model_constants[6] * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = model_constants[2] * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0;
				aux_TM = model_constants[2] * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = model_constants[14] * new_speed_vec[threadIdx.x]/*ind*/ + model_constants[13] * ohc_disp - model_constants[18] * new_ohc_psi_ind;

				new_TM_sp = model_constants[14] * new_speed_ind + model_constants[13] * ohc_disp + model_constants[18] * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*model_constants[14] - 1.0);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* model_constants[14] - 1.0;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(model_constants[25] * G_ind + 1.0);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, model_constants[0]);
				//float delta = curr_time_f* model_constants[1];
				sx[9] = curr_time_f* model_constants[1];
				//sx[10] = 1.0 - sx[9];
				sx[10] = sx[0] * (1.0 - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = model_constants[26] * ((model_constants_integers[1] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6])) + model_constants[17] * sx[8]);
				*/
				/**
				* deconstructing Y_ind = model_constants[26] * (model_constants[15] * sx[8] + model_constants[31] * sx[4] + model_constants[32] * sx[6]); to conserve registers
				*/
				Y_ind = model_constants[26] * (model_constants[32] * sx[6] + model_constants[31] * sx[4] + model_constants[15] * sx[8]);
				// also model_constants[17] * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0;
			}
			else {
				Y_ind = model_constants[9] * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				sx[9] = model_constants[31] * sx[4];
				sx[9] = model_constants[32] * sx[6] + sx[9];
				sx[10] = model_constants[15] * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = model_constants[7] * sx[10];


			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			/**
			deconstructing
			new_ohcp_deriv_ind = ((model_constants[13] * ohc_disp) + (model_constants[14] * (new_TM_sp - new_speed_vec[threadIdx.x]))) + (model_constants[18] * new_ohc_psi_ind);
			*/
			new_ohcp_deriv_ind = new_TM_sp - new_speed_ind;
			new_ohcp_deriv_ind = model_constants[14] * new_ohcp_deriv_ind;
			new_ohcp_deriv_ind = model_constants[13] * ohc_disp + new_ohcp_deriv_ind;
			new_ohcp_deriv_ind = model_constants[18] * new_ohc_psi_ind + new_ohcp_deriv_ind;
			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(model_constants[24], new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind
				fault_vec[threadIdx.x] = new_speed_ind * curr_time_step_f;
				fault_vec[threadIdx.x] = fmax(mtt, fault_vec[threadIdx.x]);
				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fault_vec[threadIdx.x];
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > model_constants[23] condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5;
						if (curr_time_step_f<model_constants[21]) {
							curr_time_step_f = model_constants[21];
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0;
						if (curr_time_step_f>model_constants[22]) {
							curr_time_step_f = model_constants[22];
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0; // another loop 
					sp_err_vec[2] = 0.0; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				//sx[11] = 1;
			}
			//if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		//if (threadIdx.x == 0) {
		// sx[13] += 1.0; // counter of iteration;
		// sx[12] += curr_time_step_f; // accumulating time steps to calculate average
		// sx[15] += sx[14];
		//}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > model_constants[20]) {
			// TODO
			curr_time_f -= model_constants[20];
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
																									//if (threadIdx.x == 0) {
																									//	Failed_Converged_Time_Node[input_sample] = sx[11];
																									//	if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
																									// float div_times = __fdividef(sx[12], sx[13]);
																									// convergence_time_measurement[input_sample] = div_times;
																									// convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
																									// float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
																									// convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
																									// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
																									// convergence_delta_time_iterations[input_sample] = sx[13];
																									// convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
																									//}
																									//__syncthreads();


																									//if (threadIdx.x == 0) {
																									//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
																									//}


																									//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}
			/**
			if (threadIdx.x == 0) {
			sx[11] = 0;
			sx[12] = 0;
			sx[13] = 0;
			sx[15] = 0;
			}*/
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}


// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, KERNEL_BLOCKS) BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Constants_kernel(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	//int * __restrict__ Failed_Converged_Time_Node,
	//int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
											   //half_curr_time_step_f = 0.5f*curr_time_step_f;
											   // time offset calculated by nodes and transfered for  float now
											   //int time_offset = nodes_per_time_block*blockIdx.x;
											   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1) ; // start from the beginng of the block
																 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																 //int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																 //int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																 //int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																 //first time to output -> starting after first overlaping period
																 //if (threadIdx.x == 0) {
																 // in case of transient write, start write from the begginning
																 // 
																 //	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																 //}


																 //first output time of next block
																 // updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																 // removed time_offset +
																 //next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																 /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																 next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																 next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																 */
																 //offset for first output sample (in units of samples in output array)
																 //int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																 //
																 //int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				
																	
				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp -w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );
				
				sx[9] = -model_a2 * sx[6] -model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);
			
			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind
				
				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				//sx[11] = 1;
			}
			//if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		//if (threadIdx.x == 0) {
		// sx[13] += 1.0f; // counter of iteration;
		// sx[12] += curr_time_step_f; // accumulating time steps to calculate average
		// sx[15] += sx[14];
		//}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
																									//if (threadIdx.x == 0) {
																									//	Failed_Converged_Time_Node[input_sample] = sx[11];
																									//	if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
																									// float div_times = __fdividef(sx[12], sx[13]);
																									// convergence_time_measurement[input_sample] = div_times;
																									// convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
																									// float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
																									// convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
																									// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
																									// convergence_delta_time_iterations[input_sample] = sx[13];
																									// convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
																									//}
																									//__syncthreads();


																									//if (threadIdx.x == 0) {
																									//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
																									//}


																									//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}
			/**
			if (threadIdx.x == 0) {
			sx[11] = 0;
			sx[12] = 0;
			sx[13] = 0;
			sx[15] = 0;
			}*/
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}
/**
* variations here will differ on kernel bounds launch blocks
*/
// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 5) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration =rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if ( threadIdx.x ==0 ) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
		 sx[13] += 1.0f; // counter of iteration;
		 sx[12] += curr_time_step_f; // accumulating time steps to calculate average
		 sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
																									//__syncthreads();


																									//if (threadIdx.x == 0) {
																									//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
																									//}


																									//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}
			
			if (threadIdx.x == 0) {
			sx[11] = 0;
			sx[12] = 0;
			sx[13] = 0;
			sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}


// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 6) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_6B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}


// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 7) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_7B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}


// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 8) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_8B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 4) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B(
	float * __restrict__ input_samples,
	float * __restrict__ saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 4) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized(
	float * __restrict__ input_samples,
	float * __restrict__ saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	float prev_OW_displacement_ind;//sx[3]
	float prev_OW_acceleration_ind; // sx[7]
	float prev_OW_speed_ind;	// sx[5]
	float new_OW_displacement_ind; // sx[4]
	float new_OW_acceleration_ind;   //sx[2]
	float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;
	float low_multiplier = threadIdx.x < SECTIONSM2 ? 1.0f : 0.0f;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	prev_OW_displacement_ind = 0.0;	  //sx[3]
	new_OW_displacement_ind = 0;			//sx[4]
	prev_OW_acceleration_ind = 0;
	new_OW_acceleration_ind = 0;
	prev_OW_speed_ind = 0;
	new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					new_OW_displacement_ind = prev_OW_speed_ind * curr_time_step_f + prev_OW_displacement_ind;
					new_OW_speed_ind = prev_OW_acceleration_ind * curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					new_OW_displacement_ind = (prev_OW_speed_ind + new_OW_speed_ind)* 0.5f*curr_time_step_f + prev_OW_displacement_ind;

					new_OW_speed_ind = (prev_OW_acceleration_ind + new_OW_acceleration_ind)* 0.5f*curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				//sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				//sx[10] = sx[0] * (1.0f - sx[9]);
				new_sample = sx[1] * delta + sx[0] * (1.0f - delta);



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind + model_Gme * new_sample);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = low_multiplier* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				//sx[9] = -model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind;
				//sx[10] = model_Gme * sx[8] + pressure_vec[1];
				//sx[10] = sx[9] + sx[10];
				float NN1 = -model_a2 * new_OW_speed_ind;
				float NN2 = model_Gme * new_sample + pressure_vec[1];
				float NN3 = -model_a1 * new_OW_displacement_ind;
				new_OW_acceleration_ind = __fdividef(NN1+NN2+NN3, sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			prev_OW_displacement_ind = new_OW_displacement_ind;
			prev_OW_speed_ind = new_OW_speed_ind;
			prev_OW_acceleration_ind = new_OW_acceleration_ind;
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 4) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_advanced_aggregations(
	float * __restrict__ input_samples,
	float * __restrict__ saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	//__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	float fault_ind = 0.0f;
	//__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	float sp_err_ind = 0.0f;

	//__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	float acc_diff_ind = 0.0f;
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	float prev_OW_displacement_ind;//sx[3]
	float prev_OW_acceleration_ind; // sx[7]
	float prev_OW_speed_ind;	// sx[5]
	float new_OW_displacement_ind; // sx[4]
	float new_OW_acceleration_ind;   //sx[2]
	float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	prev_OW_displacement_ind = 0.0;	  //sx[3]
	new_OW_displacement_ind = 0;			//sx[4]
	prev_OW_acceleration_ind = 0;
	new_OW_acceleration_ind = 0;
	prev_OW_speed_ind = 0;
	new_OW_speed_ind = 0;

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];
	float low_multiplier = threadIdx.x < SECTIONSM2 ? 1.0f : 0.0f;

	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					new_OW_displacement_ind = prev_OW_speed_ind * curr_time_step_f + prev_OW_displacement_ind;
					new_OW_speed_ind = prev_OW_acceleration_ind * curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					new_OW_displacement_ind = (prev_OW_speed_ind + new_OW_speed_ind)* 0.5f*curr_time_step_f + prev_OW_displacement_ind;

					new_OW_speed_ind = (prev_OW_acceleration_ind + new_OW_acceleration_ind)* 0.5f*curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			float new_sample; // will be sx[8]
							  //float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				//sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				//sx[10] = sx[0] * (1.0f - sx[9]);
				new_sample = sx[1] * delta + sx[0] * (1.0f - delta);



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind + model_Gme * new_sample);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = low_multiplier* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				//sx[9] = -model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind;
				//sx[10] = model_Gme * sx[8] + pressure_vec[1];
				//sx[10] = sx[9] + sx[10];
				float NN1 = -model_a2 * new_OW_speed_ind;
				float NN2 = model_Gme * new_sample + pressure_vec[1];
				float NN3 = -model_a1 * new_OW_displacement_ind;
				new_OW_acceleration_ind = __fdividef(NN1 + NN2 + NN3, sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				//acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				acc_diff_ind = fabs(new_accel_ref - new_accel_ind);
				acc_diff_ind = blockReduceSum<float, fmaxf >(acc_diff_ind);
				//sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				sp_err_ind = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				sp_err_ind = blockReduceSum<float, addition<float>  >(sp_err_ind);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				//fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				fault_ind = sp_err_ind - fmax(mtt, new_speed_ind * curr_time_step_f);
				fault_ind = blockReduceSum<float, addition<float>  >(fault_ind);
				// TODO - take only sample into account?
				//for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					//__syncthreads();
					//if (threadIdx.x<t_i) {
					//	sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
					//	fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
					//	acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					//}

				//}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_ind;	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_ind * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_ind > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sx[2] = curr_time_step_f; // broadcast result
					sx[3] = (float)another_loop;
					sx[4] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sx[2] = curr_time_step_f;
					sx[3] = 1.0f; // another loop 
					sx[4] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sx[2];
			another_loop = rint(sx[3]);
			reset_iteration = rint(sx[4]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			prev_OW_displacement_ind = new_OW_displacement_ind;
			prev_OW_speed_ind = new_OW_speed_ind;
			prev_OW_acceleration_ind = new_OW_acceleration_ind;
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 5) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations(
	float * __restrict__ input_samples, // input array
	float * __restrict__ saved_speeds, // output array
	int * __restrict__ Failed_Converged_Time_Node, // nodes that speed calculation could not converge
	int * __restrict__ Failed_Converged_Blocks, // flags for blocks that calculation could not converge
	volatile float *mass, // mass per cochlea section
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U, // uppear diagonal array, 5.2 Parallelism in the longitudinal dimension from Sabo et al master thesis
	volatile float *L, // lower diagonal array, 5.2 Parallelism in the longitudinal dimension from Sabo et al master thesis

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma, // damage to OHC per section
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc, // cut off frequency for OHC
	float time_step, // interval between nods
	float time_step_out, // interval between output nods
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time, // input start time
	float Ts, // interval between nods
	float _ohc_alpha_l, // Arbitrary constant (found by Dani Mackrants, 2007).
	float _ohc_alpha_s, // Arbitrary constant (found by Dani Mackrants, 2007).
	float model_Gme, // Coupling of oval window displacement to ear canal pressure
	float model_a0, // Boundary conditions - matrix
	float model_a1, // Boundary conditions - vector Y
	float model_a2, // Boundary conditions - vector Y
	float sigma_ow, // [gr/cm^2]Oval window aerial density
	float eta_1, // [V*sec/m] Electrical features of the hair cell.
	float eta_2, // [V/m] Electrical features of the hair cell.
	int samplesBufferLengthP1, // total output nodes, include padding
	int overlap_nodes_for_block, // per block namber of nodes processed to dissipate transient effect
	float cuda_min_time_step,
	float cuda_max_time_step
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	float rsm_ind;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	__shared__ float sx[SX_SIZE];
	
	float3 acc_sp_fault_ind; // combined acc, sp and fault summary for accumulations
	acc_sp_fault_ind.x = 0.0f;
	acc_sp_fault_ind.y = 0.0f;
	acc_sp_fault_ind.z = 0.0f;
	float P_tm_ind;
	float new_disp_ind;
	float new_speed_ind = 0.0f;
	float curr_time_step_f;
	float curr_time_f;
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	float prev_OW_displacement_ind;//sx[3]
	float prev_OW_acceleration_ind; // sx[7]
	float prev_OW_speed_ind;	// sx[5]
	float new_OW_displacement_ind; // sx[4]
	float new_OW_acceleration_ind;   //sx[2]
	float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	prev_OW_displacement_ind = 0.0;	  //sx[3]
	new_OW_displacement_ind = 0;			//sx[4]
	prev_OW_acceleration_ind = 0;
	new_OW_acceleration_ind = 0;
	prev_OW_speed_ind = 0;
	new_OW_speed_ind = 0;

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	
	new_accel_ind = 0.0f;

	rsm_ind = rsM[threadIdx.x];
	
	pressure_vec[threadIdx.x + 1] = 0.0f;
	

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];
	float low_multiplier = threadIdx.x < SECTIONSM2 ? 1.0f : 0.0f;

	curr_time_step_f = time_step;	   //time_step
									   
	curr_time_f = 0.0f; // cahnged to be base on time step



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										
																										
	int another_loop = 1;
	// read next input to shared memory and sychronize
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	// while input processing did not reached end of data processed in this block
	while (input_sample<mesi) {



		__syncthreads();

		
		// First Step - make approximation using EULER/MEULER

		// this loop while convergence did not reached and minimum time frame didnt reached either
		for (int out_loop = 0, reset_iteration = 1; another_loop;) {

			// copy previous speed and accelaration for reference
			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;
			// if iteration reset or first convergence
			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;
				// TM displacement
				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				// thread 0 calculate contemporary oval window displacement and speed
				if (threadIdx.x == 0) {
					new_OW_displacement_ind = prev_OW_speed_ind * curr_time_step_f + prev_OW_displacement_ind;
					new_OW_speed_ind = prev_OW_acceleration_ind * curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  - since its not first iteration, interpolate base on current and previous iteration deriviates
				// interpolate accelaration for speed
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;
				// interpolate speed for placement
				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					// Oval window displacement & speed  - since its not first iteration, interpolate base on current and previous iteration deriviates
					new_OW_displacement_ind = (prev_OW_speed_ind + new_OW_speed_ind)* 0.5f*curr_time_step_f + prev_OW_displacement_ind;

					new_OW_speed_ind = (prev_OW_acceleration_ind + new_OW_acceleration_ind)* 0.5f*curr_time_step_f + prev_OW_speed_ind;
				}
				__syncthreads();
				// TM displacement - since its not first iteration, interpolate base on current and previous iteration deriviates
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				// BMvoltage and its dereviate  - since its not first iteration, interpolate base on current and previous iteration deriviates
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				
				// N_TM_sp replaced by 	new_TM_sp to converse registers
				
				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl


			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			float new_sample; // will be sx[8]
							  //float _bc;

			if (threadIdx.x == 0) {
				
				// relative position of x in the interval
				float delta = __fdividef(curr_time_f, Ts);
				// interpolate linearily between current and next sample to find sample at time between samples
				new_sample = sx[1] * delta + sx[0] * (1.0f - delta);



				/*
				this is equation of Y_0 from sabo et al thesis (unnumbered) under Eq. 3.5
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind + model_Gme * new_sample);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				// boundary condition cochlear last section short circuited
				Y_ind = 0.0f;
			}
			else {
				// derived from Sabo et al Eq. 3.7
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			// calculating jacobby relaxation - Eq. 3.12 from Sabo et al
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				// calculates next converged pressure
				float tmp_p = low_multiplier* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 
				// updates data and sync

				__syncthreads();


				// write updated data

				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration - oval window variable calculated on thread 0 only
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				float NN1 = -model_a2 * new_OW_speed_ind;
				float NN2 = model_Gme * new_sample + pressure_vec[1];
				float NN3 = -model_a1 * new_OW_displacement_ind;
				new_OW_acceleration_ind = __fdividef(NN1 + NN2 + NN3, sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//////////////////////////// TESTER OF LIPHSCHITZ CRITERIA //////////////////////////////


			// find the speed error (compared to the past value)
			if (reset_iteration == 0) // if iteration was not reset, test another loop
			{

				acc_sp_fault_ind.x = fabs(new_accel_ref - new_accel_ind);
				acc_sp_fault_ind.y = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				//fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				acc_sp_fault_ind.z  = acc_sp_fault_ind.y - fmax(mtt, new_speed_ind * curr_time_step_f);
				//fault_ind = blockReduceSum<float, addition<float>  >(fault_ind);
				// test for faulty speed convergence, if not every section succeeds they all fail
				// this is testing for liphschitz aggregation  criteria
				acc_sp_fault_ind = blockReduceTripleAggregators<fmaxf, addition<float>, addition<float> >(acc_sp_fault_ind);
				// TODO - take only sample into account?
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = acc_sp_fault_ind.y;	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_sp_fault_ind.x * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass convergence of summary failed , E_ref_mean in section 6.1 item 2 in Sabo at al size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (acc_sp_fault_ind.z > 0) {
						// another iteration is needed for this step - due to at least on section failed to converge
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sx[2] = curr_time_step_f; // broadcast result
					sx[3] = (float)another_loop;
					sx[4] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {
				// first iteration, ensures another loop
				if (threadIdx.x < 32) {
					sx[2] = curr_time_step_f;
					sx[3] = 1.0f; // another loop 
					sx[4] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();

			// updates time step for next iteration
			curr_time_step_f = sx[2];
			another_loop = rint(sx[3]);
			reset_iteration = rint(sx[4]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		// ensure atleast one loop next iteration
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		// need to process next sample
		if (curr_time_f > time_step_out) {
			// update relative time between samples to be remainder of time step, view Yonatan Koral Thesis Eq. 4.27-4.28
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// since part of input samples processing do not written to output due to transient effects, here we test if passed transient effects - time > t_BOP from Fig 4.2 in Yonatan Koral Thesis
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				// output convergence tests results
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past

		// copy current speeds,displacements and accelaration to previous ones for next iteration

		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		// oival window copying done for thread 0 only
		if (threadIdx.x == 0) {
			prev_OW_displacement_ind = new_OW_displacement_ind;
			prev_OW_speed_ind = new_OW_speed_ind;
			prev_OW_acceleration_ind = new_OW_acceleration_ind;
		}

		__syncthreads();
		// current time updated
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 3) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_3B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 2) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_2B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

// BM calculation kernel, while less finely divided on cochlea property consume less memory
// this versions remove self diagnostic but returns pre fmaf computations and avoid use constants array, all data transfered trough parameters
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 1) BMOHC_FAST_Pre_fmaf_No_Constants_kernel_1B(
	float * __restrict__ input_samples,
	volatile float *saved_speeds,
	int * __restrict__ Failed_Converged_Time_Node,
	int * __restrict__ Failed_Converged_Blocks,
	volatile float *mass,
	volatile float *rsM, // reciprocal mass for multiplication instead of division
	volatile float *U,
	volatile float *L,

	volatile float *R,
	volatile float *S,
	volatile float *Q,

	volatile float *gamma,
	volatile float *S_ohc,
	volatile float *S_tm,
	volatile float *R_tm,
	float * __restrict__ gen_model_throw_tolerance,
	float * __restrict__ gen_model_max_m1_sp_tolerance,
	int * __restrict__ gen_model_out_sample_index,
	int * __restrict__ gen_model_end_sample_index,
	float * __restrict__ convergence_time_measurement,
	float * __restrict__ convergence_time_measurement_blocks,
	float * __restrict__ convergence_delta_time_iterations,
	float * __restrict__ convergence_delta_time_iterations_blocks,
	float * __restrict__ convergence_jacoby_loops_per_iteration,
	float * __restrict__ convergence_jacoby_loops_per_iteration_blocks,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	float base_time,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int samplesBufferLengthP1,
	int overlap_nodes_for_block,
	float cuda_min_time_step,
	float cuda_max_time_step
	//float * __restrict__ convergence_time_measurement,
	//float * __restrict__ convergence_time_measurement_blocks,
	//float * __restrict__ convergence_delta_time_iterations,
	//float * __restrict__ convergence_delta_time_iterations_blocks,
	//float * __restrict__ convergence_jacoby_loops_per_iteration,
	//float * __restrict__ convergence_jacoby_loops_per_iteration_blocks
) {

	int mosi = gen_model_out_sample_index[blockIdx.x];
	int mesi = gen_model_end_sample_index[blockIdx.x];
	float mtt = gen_model_throw_tolerance[blockIdx.x];
	float mmspt = gen_model_max_m1_sp_tolerance[blockIdx.x];
	//__shared__ float l_vec[FIRST_STAGE_MEMSIZE]; // for Jaccoby, replaced with threadIdx.x<SECTIONS-2
	//__shared__ float u_vec[FIRST_STAGE_MEMSIZE];  // tesing looks like this is always 1 so...
	//__shared__ float rsm_vec[FIRST_STAGE_MEMSIZE];
	float rsm_ind;
	//float l_value;
	//float u_value;
	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];

	//__shared__ float curr_time[WARP_SIZE];
	//__shared__ float curr_time_step[WARP_SIZE];
	//__shared__ float half_curr_time_step[WARP_SIZE];
	__shared__ float sx[SX_SIZE];
	//if (threadIdx.x == 0) {
	// sx[11] = 0;
	//	Failed_Converged_Blocks[blockIdx.x] = 0;
	// convergence_time_measurement_blocks[blockIdx.x] = 0;
	// convergence_delta_time_iterations_blocks[blockIdx.x] = 0;
	// convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] = 0;
	//}
	//__shared__ int common_ints[COMMON_INTS_SIZE];
	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	//__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	float P_tm_ind;
	//__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	float new_disp_ind;
	//__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];
	float new_speed_ind = 0.0f;
	//float new_speed_value;
	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	//__shared__ float common_args_vec[1]; 
	float curr_time_step_f;
	//float half_curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f;  
	//float next_block_first_sample_f; 
	float S_bm_ind;
	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	//float prev_OW_displacement_ind;//sx[3]
	//float prev_OW_acceleration_ind; // sx[7]
	//float prev_OW_speed_ind;	// sx[5]
	//float new_OW_displacement_ind; // sx[4]
	//float new_OW_acceleration_ind;   //sx[2]
	//float new_OW_speed_ind; // sx[6]
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	//float mass_ind;
	float reciprocal_mass_ind; // for faster calculations
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;

	//int wrap_id = threadIdx.x >> 5;


	prev_TM_disp_ind = 0.0f;
	new_TM_disp = 0.0f;
	prev_TM_sp_ind = 0.0f;
	new_TM_sp = 0.0f;

	//prev_OW_displacement_ind = 0.0;	  //sx[3]
	//new_OW_displacement_ind = 0;			//sx[4]
	//prev_OW_acceleration_ind = 0;
	//new_OW_acceleration_ind = 0;
	//prev_OW_speed_ind = 0;
	//new_OW_speed_ind = 0;   

	prev_disp_ind = 0.0f;
	prev_speed_ind = 0.0f;
	prev_accel_ind = 0.0f;

	new_disp_ind = 0.0f;
	//new_disp_vec[threadIdx.x] = 0.0f;
	//new_speed_ind = 0; 
	//new_speed_vec[threadIdx.x] = 0; 
	//new_speed_vec[threadIdx.x] = 0.0f; => already loaded
	new_accel_ind = 0.0f;

	//rsm_vec[threadIdx.x] = rsM[threadIdx.x];
	rsm_ind = rsM[threadIdx.x];
	//u_vec[threadIdx.x] = U[threadIdx.x];
	//l_vec[threadIdx.x] = L[threadIdx.x];
	//u_value = U[threadIdx.x];
	//l_value = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0f;
	/*
	if (threadIdx.x==0)
	{
	pressure_vec[0] = 0.0;
	pressure_vec[SECTIONS+1] = 0.0;
	m_vec[0] = 0.0;
	m_vec[SECTIONS + 1] = 0.0;
	u_vec[0] = 0.0;
	u_vec[SECTIONS + 1] = 0.0;
	l_vec[0] = 0.0;
	l_vec[SECTIONS + 1] = 0.0;
	}
	*/

	if (threadIdx.x < SX_SIZE) sx[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) {
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0f;
		sx[15] = 0;
	}
	__syncthreads();
	prev_ohc_psi_ind = 0.0f;
	prev_ohcp_deriv_ind = 0.0f;
	new_ohc_psi_ind = 0.0f;



	//mass_ind = mass[threadIdx.x]; 
	reciprocal_mass_ind = 1.0f / mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	//S_vec[threadIdx.x] = S[threadIdx.x];
	S_bm_ind = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];


	curr_time_step_f = time_step;	   //time_step
									   //half_curr_time_step_f = 0.5f*curr_time_step_f;
									   // time offset calculated by nodes and transfered for  float now
									   //int time_offset = nodes_per_time_block*blockIdx.x;
									   //curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
					   //float_base_time_f = 0.0f;
					   //int the_start_block = base_index == 0 && blockIdx.x == 0 ? 1 : 0;


					   //if (threadIdx.x == 0) {
					   //int transient_offset = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   //  common_ints[0] = (blockIdx.x > 0 && (model_constants_integers[0] == 0 || (model_constants_integers[0] != 1 && blockIdx.x%model_constants_integers[0] != 0)));
					   // isDecoupled = blockIdx.x%isDecoupled; 
					   // isDecoupled = isDecoupled > 0 ? 0 : 1;
					   // int preDecoupled
					   //  common_ints[1] = model_constants_integers[0]>0 && (model_constants_integers[0] == 1 || ((blockIdx.x + 1) % model_constants_integers[0] == 0)) && blockIdx.x != gridDim.x - 1; // divide but not last block of this interval, that one will be handeled outside

					   //}
					   // if (blockIdx.x > 0 && (Decouple_Filter == 0 || (Decouple_Filter != 1 && blockIdx.x%Decouple_Filter!=0))) transient_offset = 0; // transient will not shown after first block
					   // Main Algorithm
					   // will be 1 if the next block will be decoupled from this block
					   //int preDecoupled = Decouple_Filter;
					   // will be one if this block is decoupled from last block
					   //int isDecoupled = Decouple_Filter;



	int input_sample = (samplesBufferLengthP1 - overlap_nodes_for_block)* blockIdx.x / (gridDim.x + 1); // start from the beginng of the block
																										//int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
																										//int out_sample = input_sample + (1 - transient_offset)*model_constants_integers[5]; // use as start output
																										//int out_sample = model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE];
																										//int end_output = input_sample + model_constants_integers[8] + (1 - preDecoupled)*model_constants_integers[5];   // now model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE_P1]
																										//first time to output -> starting after first overlaping period
																										//if (threadIdx.x == 0) {
																										// in case of transient write, start write from the begginning
																										// 
																										//	common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
																										//}


																										//first output time of next block
																										// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
																										// removed time_offset +
																										//next_block_first_sample_f = float(( nodes_per_time_block+overlap_nodes_for_block+1)*time_step_out); // cahnged to be base on time step
																										/*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										*/
																										//offset for first output sample (in units of samples in output array)
																										//int time_offset = rint((common_args_vec[0]-base_time)/time_step_out);
																										//
																										//int time_offset = rint((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	//int oloop_limit =  _CUDA_OUTERN_LOOPS; 
	//int Lipschits_en = 1; // unecessary removed



	//float sx1 = input_samples[input_sample];
	//float sx2 = input_samples[input_sample+1];
	if (threadIdx.x <2) {
		sx[threadIdx.x] = input_samples[input_sample + threadIdx.x];
	}
	__syncthreads();
	//float P_TM;
	//float m_inv = 1.0/M[threadIdx.x];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	while (input_sample<mesi) {



		__syncthreads();

		// if (Lipschits_en) oloop_limit =  _CUDA_OUTERN_LOOPS; 


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop;) {


			float new_speed_ref = new_speed_ind; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_sp_ind * curr_time_step_f + prev_TM_disp_ind;

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_ind/*ind*/ = prev_accel_ind * curr_time_step_f + prev_speed_ind;


				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {
					sx[4] = sx[5] * curr_time_step_f + sx[3];
					sx[6] = sx[7] * curr_time_step_f + sx[5];
				}
				__syncthreads();
				// OHC:  
				new_ohc_psi_ind = prev_ohcp_deriv_ind* curr_time_step_f + prev_ohc_psi_ind;



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_ind/*ind*/ = (prev_accel_ind + new_accel_ind)* 0.5f*curr_time_step_f + prev_speed_ind;

				new_disp_ind = (prev_speed_ind + new_speed_ind)* 0.5f*curr_time_step_f + prev_disp_ind;
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );


				// model_constants_integers[1] && - assuming enbable_OW always active
				if (threadIdx.x == 0) {

					sx[4] = (sx[5] + sx[6])* 0.5f*curr_time_step_f + sx[3];

					sx[6] = (sx[7] + sx[2])* 0.5f*curr_time_step_f + sx[5];
				}
				__syncthreads();
				new_TM_disp = (prev_TM_sp_ind + new_TM_sp)* 0.5f*curr_time_step_f + prev_TM_disp_ind;

				// OHC: 
				new_ohc_psi_ind = (prev_ohcp_deriv_ind + new_ohcp_deriv_ind)* 0.5f*curr_time_step_f + prev_ohc_psi_ind;

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_ind; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			if (out_loop<CUDA_AUX_TM_LOOPS) {
				float deltaL_disp;
				float aux_TM;
				//if (model_constants_integers[2]) // Non Linear PSI model - assuming always active
				//{
				float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


				aux_TM = tanhf(tan_arg);
				deltaL_disp = -1.0f*_ohc_alpha_s * aux_TM;

				//Calc_TM_speed
				//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;

				//}
				//else
				//{
				//	deltaL_disp = -1.0f*_ohc_alpha_l * new_ohc_psi_ind;

				//Calc_TM_speed
				//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
				//	aux_TM = (__fdividef(_ohc_alpha_l,_ohc_alpha_s)) * new_ohc_psi_ind;
				//   aux_TM = aux_TM*aux_TM;

				//}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				//aux_TM = _ohc_alpha_l * (aux_TM - 1.0);
				aux_TM = aux_TM* aux_TM - 1.0f;
				aux_TM = _ohc_alpha_l * aux_TM;


				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				//float N11;
				//float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;

				// N_TM_sp replaced by 	new_TM_sp to converse registers
				//new_TM_sp = eta_2 * new_speed_vec[threadIdx.x]/*ind*/ + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;

				new_TM_sp = eta_2 * new_speed_ind + eta_1 * ohc_disp - w_ohc * new_ohc_psi_ind;
				new_TM_sp = S_tm_ind* deltaL_disp + new_TM_sp*R_tm_ind*aux_TM;

				// P_TM_vec temporary used here to conserve registers, sorry for the mess
				// P_TM_vec[threadIdx.x] -> P_tm_ind 
				P_tm_ind = S_tm_ind* new_TM_disp + new_ohc_pressure_ind;
				new_TM_sp = P_tm_ind - gamma_ind* new_TM_sp;
				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				//float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0f);
				// D_TM_Sp will replaced by aux_TM to conserve registers...
				aux_TM = gamma_ind*aux_TM* eta_2 - 1.0f;
				aux_TM = R_tm_ind*aux_TM;
				new_TM_sp = __fdividef(new_TM_sp, aux_TM);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;
				P_tm_ind = R_tm_ind* new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			//float R_bm_nl;  replaced by G_ind to converse registers
			//if (alpha_r == 0.0)
			//	R_bm_nl = R_ind;
			//else

			float G_ind = new_speed_ind * new_speed_ind;
			G_ind = R_ind*(alpha_r * G_ind + 1.0f);

			G_ind = G_ind* new_speed_ind/*ind*/ + S_bm_ind * new_disp_ind/*ind*/;
			G_ind = -G_ind - P_tm_ind;
			//G_ind = G_ind + P_tm_ind;
			//G_ind = -1.0f*G_ind;
			//float dx_pow2 = delta_x * delta_x;


			float Y_ind;

			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			//float new_sample; // will be sx[8]
			//float _bc;

			if (threadIdx.x == 0) {
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				//float delta = __fdividef(curr_time_f, Ts);
				//float delta = curr_time_f* Fs;
				//sx[9] = curr_time_f* Fs;
				sx[9] = __fdividef(curr_time_f, Ts);
				//sx[10] = 1.0f - sx[9];
				sx[10] = sx[0] * (1.0f - sx[9]);
				sx[8] = sx[1] * sx[9] + sx[10];



				/*
				if (enable_OW)
				{
				_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
				_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				now I can avoid control
				however since enable_OW = model_constants_integers[1] is always one lets short the formula
				Y_ind = delta_x * model_a0 * ((model_constants_integers[1] * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6])) + (1 - enable_OW)*model_Gme * sx[8]);
				*/
				/**
				* deconstructing Y_ind = delta_x * model_a0 * (model_Gme * sx[8] -model_a1 * sx[4] -model_a2 * sx[6]); to conserve registers
				*/
				Y_ind = delta_x * model_a0 * (-model_a2 * sx[6] - model_a1 * sx[4] + model_Gme * sx[8]);
				// also (1 - enable_OW)*model_Gme * sx[8] is removed since enable_OW is always one
				//Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1)) {
				Y_ind = 0.0f;
			}
			else {
				Y_ind = delta_x*delta_x * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}
			__syncthreads();
			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? model_constants_integers[10] : model_constants_integers[9];
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++) {
				__syncthreads();
				float tmp_p = float(threadIdx.x<SECTIONSM2)* pressure_vec[threadIdx.x] + pressure_vec[threadIdx.x + 2];
				tmp_p = Y_ind - tmp_p;
				// u_vec  is all 1's so it's removed
				//l_vec[threadIdx.x] replaced by threadIdx.x<SECTIONSM2
				tmp_p = tmp_p*rsm_ind; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}

			__syncthreads();

			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			//new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);
			new_accel_ind = pressure_vec[threadIdx.x + 1] + G_ind;
			new_accel_ind = new_accel_ind* reciprocal_mass_ind;

			//__syncthreads(); 

			// assuming model_constants_integers[1] && enable_OW always active
			if (threadIdx.x == 0) {

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );

				sx[9] = -model_a2 * sx[6] - model_a1 * sx[4];
				sx[10] = model_Gme * sx[8] + pressure_vec[1];
				sx[10] = sx[9] + sx[10];
				sx[2] = __fdividef(sx[10], sigma_ow);

			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;
			// __syncthreads();
			// return from deconstructing
			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_ind))) + (-w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)
			// rempved Lipschits_en due to unecessacity	
			if (reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				//float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = fabs(new_speed_ind/*ind*/ - new_speed_ref);
				//float sp_err_limit = ;
				// deconstructing (sp_err_vec[threadIdx.x]>fmax(Max_Tolerance_Parameter, new_speed_vec[threadIdx.x] * curr_time_step_f)) ? 1 : 0;
				// fault_vec => fault_ind

				fault_vec[threadIdx.x] = sp_err_vec[threadIdx.x] - fmax(mtt, new_speed_ind * curr_time_step_f);
				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
					__syncthreads();
					if (threadIdx.x<t_i) {
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				//int fault_int = __syncthreads_or(fault_ind > 0.0f);
				__syncthreads();

				// calculate lipschitz number 
				if (threadIdx.x<32) {
					float m1_sp_err = sp_err_vec[0];	 // shared on single warp not necessary
					float Lipschitz;  // shared on single warp replaced by 	sp_err_vec[1] to conserve registers
									  // Lipschitz calculations (threads 0-32)
					if (m1_sp_err > mmspt) {
						Lipschitz = acc_diff_vec[0] * curr_time_step_f;
						Lipschitz = __fdividef(Lipschitz, m1_sp_err);

					}
					else
						Lipschitz = 0.0f;

					//float Lmul = sp_err_vec[1] * curr_time_step_f;   // replaced by sp_err_vec[1] to conserve registers (multiplication moved into the   sp_err_vec[0] > Max_M1_SP_Error_Parameter condition

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lipschitz > 6.0f) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5f;
						if (curr_time_step_f<cuda_min_time_step) {
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0) {
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lipschitz < 0.5f)//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0f;
						if (curr_time_step_f>cuda_max_time_step) {
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else {
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32





			} // !first_iteration
			else {

				if (threadIdx.x < 32) {
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0f; // another loop 
					sp_err_vec[2] = 0.0f; // reset_iteration 
				}

				//another_loop = 1;	 
				//reset_iteration = 0;
			}


			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = rint(sp_err_vec[1]);
			reset_iteration = rint(sp_err_vec[2]);

			//half_curr_time_step_f = 0.5f*curr_time_step_f;

			/////////////////////////////////////////////////////////



			out_loop++;
			// oloop_limit replaced by _CUDA_OUTERN_LOOPS due to unecessaity
			// out_loop >= _CUDA_OUTERN_LOOPS
			// convergence has failed
			if (out_loop >= model_constants_integers[11]) {
				another_loop = 0;
				reset_iteration = 1;
				sx[11] = 1;
			}
			if (threadIdx.x == 0) sx[14] = float(out_loop);

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;
		if (threadIdx.x == 0) {
			sx[13] += 1.0f; // counter of iteration;
			sx[12] += curr_time_step_f; // accumulating time steps to calculate average
			sx[15] += sx[14];
		}
		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out) {
			// TODO
			curr_time_f -= time_step_out;
			if (threadIdx.x == 0) {
				sx[0] = sx[1];
				sx[1] = input_samples[input_sample + 2];	 // input sample updated later so its +2 instead of +1
			}
			__syncthreads();
			// out_sample replaced by model_constants_integers[(blockIdx.x<<1) + MODEL_INTEGERS_CONSTANTS_SIZE] due to irrelevancy
			if (input_sample + 1 >= mosi) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				//int t_ind = ;


				//saved_speeds[SECTIONS*(input_sample - 1) + threadIdx.x] = new_speed_ind/*ind*/;
				saved_speeds[(input_sample << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// -1 removed due to input sample position updated later and now I can do bitwise ops...
				if (threadIdx.x == 0) {
					Failed_Converged_Time_Node[input_sample] = sx[11];
					if (sx[11] > 0) Failed_Converged_Blocks[blockIdx.x] += 1;
					float div_times = __fdividef(sx[12], sx[13]);
					convergence_time_measurement[input_sample] = div_times;
					convergence_time_measurement_blocks[blockIdx.x] += __fdividef(div_times, float(mesi - mosi));
					float div_jacoby_iterations = __fdividef(sx[15], sx[13]);
					convergence_jacoby_loops_per_iteration[input_sample] = div_jacoby_iterations;
					convergence_jacoby_loops_per_iteration_blocks[blockIdx.x] += __fdividef(div_jacoby_iterations, float(mesi - mosi));
					convergence_delta_time_iterations[input_sample] = sx[13];
					convergence_delta_time_iterations_blocks[blockIdx.x] += __fdividef(sx[13], float(mesi - mosi));
				}
				//__syncthreads();


				//if (threadIdx.x == 0) {
				//	common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				//}


				//out_sample++;


			}
			else {
				saved_speeds[(mosi << LOG_SECTIONS) | threadIdx.x] = new_speed_ind/*ind*/;	// alternate target to force synchronization, -1 removed due to input sample position updated later and now I can do bitwise ops...
			}

			if (threadIdx.x == 0) {
				sx[11] = 0;
				sx[12] = 0;
				sx[13] = 0;
				sx[15] = 0;
			}
			input_sample++;
			//__syncthreads(); // NEW NEW 
			//if (threadIdx.x == 0) {
			//}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_ind; //ind;
		prev_speed_ind = new_speed_ind/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;

		// model_constants_integers[1] && enable_OW is always 1
		if (threadIdx.x == 0) {
			sx[3] = sx[4];
			sx[5] = sx[6];
			sx[7] = sx[2];
		}

		__syncthreads();
		curr_time_f += curr_time_step_f;




	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}

/**
* this variation of the function copied from backup on 2017 January, reflected original structure of Doron Program
* should not run, compiled to test registers number only, launch bounds max block changed to 4, and naming the kernel to avoid duplicate names
* are the only changes apply for this kernel
*/
__global__ void LAUNCHBOUNDS(FIRST_STAGE_MEMSIZE, 4) BMOHC_OLD_2017_01_13_kernel(
	float *input_samples,
	float *saved_speeds,
	float *mass,
	float *M,
	float *U,
	float *L,

	float *R,
	float *S,
	float *Q,

	float *gamma,
	float *S_ohc,
	float *S_tm,
	float *R_tm,
	float w_ohc,
	float time_step,
	float time_step_out,
	float delta_x,
	float alpha_r,
	int enable_psi,
	int enable_OW,
	int base_index,
	float Ts,
	float _ohc_alpha_l,
	float _ohc_alpha_s,
	float model_Gme,
	float model_a0,
	float model_a1,
	float model_a2,
	float sigma_ow,
	float eta_1,
	float eta_2,
	int nodes_per_time_block,
	int overlap_nodes_for_block,
	long overlapTimeMicroSec,
	int show_transient,
	float cuda_max_time_step,
	float cuda_min_time_step
)
{


	__shared__ float l_vec[FIRST_STAGE_MEMSIZE + 2]; // for Jaccoby
	__shared__ float u_vec[FIRST_STAGE_MEMSIZE + 2];
	__shared__ float m_vec[FIRST_STAGE_MEMSIZE + 2];

	__shared__ float pressure_vec[FIRST_STAGE_MEMSIZE + 2];



	__shared__ float fault_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float sp_err_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float acc_diff_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float S_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float P_TM_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float new_disp_vec[FIRST_STAGE_MEMSIZE];
	__shared__ float new_speed_vec[FIRST_STAGE_MEMSIZE];

	// copy all arrays to shared memory - to save global memory accesses when processing multiple points     
	__shared__ float common_args_vec[1];
	float curr_time_step_f;
	float curr_time_f;
	//float float_base_time_f = 0.0f;
	//float next_block_first_sample_f =0.0f;

	float new_ohcp_deriv_ind;
	float new_ohc_psi_ind;
	float new_TM_disp;
	float new_TM_sp;

	float prev_disp_ind;
	float prev_speed_ind;
	float prev_accel_ind;
	float prev_ohc_psi_ind;
	float prev_ohcp_deriv_ind;
	float prev_OW_displacement_ind;
	float prev_OW_acceleration_ind;
	float prev_OW_speed_ind;
	float new_OW_displacement_ind;
	float new_OW_acceleration_ind;
	float new_OW_speed_ind;
	float prev_TM_disp_ind;
	float prev_TM_sp_ind;




	float new_ohc_pressure_ind;



	float gamma_ind;

	float R_ind;

	//float new_disp_ind;
	//float new_speed_ind;
	float new_accel_ind;



	float Q_ind;
	float mass_ind;
	float S_ohc_ind;
	float S_tm_ind;
	float R_tm_ind;
	float Y_ind;




	prev_TM_disp_ind = 0.0;
	new_TM_disp = 0.0;
	prev_TM_sp_ind = 0.0;
	new_TM_sp = 0.0;

	prev_OW_displacement_ind = 0.0;
	new_OW_displacement_ind = 0;
	prev_OW_acceleration_ind = 0;
	new_OW_acceleration_ind = 0;
	prev_OW_speed_ind = 0;
	new_OW_speed_ind = 0;

	prev_disp_ind = 0;
	prev_speed_ind = 0;
	prev_accel_ind = 0;

	//new_disp_ind = 0;
	new_disp_vec[threadIdx.x] = 0;
	//new_speed_ind = 0; 
	new_speed_vec[threadIdx.x] = 0;
	new_accel_ind = 0;

	m_vec[threadIdx.x + 1] = M[threadIdx.x];
	u_vec[threadIdx.x + 1] = U[threadIdx.x];
	l_vec[threadIdx.x + 1] = L[threadIdx.x];
	pressure_vec[threadIdx.x + 1] = 0.0;
	if (threadIdx.x == 0)
	{
		pressure_vec[0] = 0.0;
		pressure_vec[SECTIONS + 1] = 0.0;
		m_vec[0] = 0.0;
		m_vec[SECTIONS + 1] = 0.0;
		u_vec[0] = 0.0;
		u_vec[SECTIONS + 1] = 0.0;
		l_vec[0] = 0.0;
		l_vec[SECTIONS + 1] = 0.0;
	}


	prev_ohc_psi_ind = 0.0;
	prev_ohcp_deriv_ind = 0.0;
	new_ohc_psi_ind = 0.0;



	mass_ind = mass[threadIdx.x];
	gamma_ind = gamma[threadIdx.x];
	R_ind = R[threadIdx.x];
	S_vec[threadIdx.x] = S[threadIdx.x];
	Q_ind = Q[threadIdx.x];
	S_ohc_ind = S_ohc[threadIdx.x];
	S_tm_ind = S_tm[threadIdx.x];
	R_tm_ind = R_tm[threadIdx.x];

	__syncthreads();

	int transient_offset = show_transient;
	if (blockIdx.x > 0) transient_offset = 0; // transient will not shown after first block
											  // Main Algorithm
	curr_time_step_f = time_step;
	// time offset calculated by nodes and transfered for  float now
	//int time_offset = nodes_per_time_block*blockIdx.x;
	//curr_time_f = base_time + ((float)(blockIdx.x*time_block_length_extended));
	curr_time_f = 0.0f;// float(time_offset*time_step_out); // cahnged to be base on time step
	//float_base_time_f = 0.0f;
	int input_sample = base_index + nodes_per_time_block*blockIdx.x; // start from the beginng of the block
																	 //int start_output = input_sample + (show_transient - transient_offset)*overlap_nodes_for_block;
	int out_sample = input_sample + (1 - transient_offset)*overlap_nodes_for_block; // use as start output
	int end_output = input_sample + nodes_per_time_block + overlap_nodes_for_block;
	//first time to output -> starting after first overlaping period
	if (threadIdx.x == 0) {
		// in case of transient write, start write from the begginning
		// 
		common_args_vec[0] = curr_time_f + (1 - transient_offset)*float(overlapTimeMicroSec) / 1000000.0f;
	}


	//first output time of next block
	// updated mode calculated next block from base time directly to ensure full time alignment with base time to fix quantization errors, calculating with resolution of sampling
	// removed time_offset +
	//next_block_first_sample_f = float((nodes_per_time_block + overlap_nodes_for_block + 1)*time_step_out); // cahnged to be base on time step
																										   /*	next_block_first_sample_f calculated from combine of curr time properly computed + length of block + overlapping in seconds units
																										   next_block_first_sample_f = curr_time_f + time_block_length_extended_overlapped;
																										   next_block_first_sample_f = next_block_first_sample_f + time_step_out;
																										   */
																										   //offset for first output sample (in units of samples in output array)
																										   //int time_offset = (int)((common_args_vec[0]-base_time)/time_step_out);
																										   //
																										   //int time_offset = (int)((curr_time_f - base_time) / time_step_out);





	int another_loop = 1;


	int oloop_limit = _CUDA_OUTERN_LOOPS;
	int Lipschits_en = 1;



	float sx1 = input_samples[input_sample];
	float sx2 = input_samples[input_sample + 1];


	//float P_TM;
	float m_inv = 1.0 / m_vec[threadIdx.x + 1];

	// curr_time_f<next_block_first_sample_f replaced with out_sample measurement for prcision
	// previous test bound curr_time_f<next_block_first_sample_f
	// (out_sample<nodes_per_time_block)
	// curr_time_f<next_block_first_sample_f
	for (int sample_loop = 0; input_sample<end_output; sample_loop++)
	{



		__syncthreads();

		if (Lipschits_en) oloop_limit = _CUDA_OUTERN_LOOPS;


		// First Step - make approximation using EULER/MEULER


		for (int out_loop = 0, reset_iteration = 1; another_loop; )
		{


			float new_speed_ref = new_speed_vec[threadIdx.x]; //ind;
			float new_accel_ref = new_accel_ind;

			if ((out_loop == 0) || reset_iteration) // first iteration -> EULER
			{
				out_loop = 0;

				new_TM_disp = prev_TM_disp_ind + (prev_TM_sp_ind*curr_time_step_f);

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				//new_disp_vec[threadIdx.x]	= prev_disp_p+(prev_speed_p*curr_time_step_f ); 
				new_speed_vec[threadIdx.x]/*ind*/ = prev_speed_ind + (prev_accel_ind*curr_time_step_f);


				new_disp_vec[threadIdx.x] = prev_disp_ind + (0.5*(prev_speed_ind + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f);

				if (enable_OW && threadIdx.x == 0)
				{
					new_OW_displacement_ind = prev_OW_displacement_ind + (prev_OW_speed_ind*curr_time_step_f);
					new_OW_speed_ind = prev_OW_speed_ind + (prev_OW_acceleration_ind*curr_time_step_f);
				}

				// OHC:  
				new_ohc_psi_ind = prev_ohc_psi_ind + (prev_ohcp_deriv_ind*curr_time_step_f);



			}
			else		//  TRAPEZOIDAL 
			{

				// BM displacement & speed  
				//new_disp_ind	= prev_disp_ind+(0.5*(prev_speed_p + new_speed_vec[threadIdx.x]/*ind*/)*curr_time_step_f ); 
				new_speed_vec[threadIdx.x]/*ind*/ = prev_speed_ind + (0.5*(prev_accel_ind + new_accel_ind)*curr_time_step_f);

				new_disp_vec[threadIdx.x] = prev_disp_ind + (0.5*(prev_speed_ind + new_speed_vec[threadIdx.x])*curr_time_step_f);
				// not enough shared mem for trapezoidal 
				// new_speed_vec[threadIdx.x]	= prev_speed_vec[tx]+(0.5*(prev_accel_vec[threadIdx.x] + new_accel_vec[threadIdx.x])*curr_time_step_f );



				if (enable_OW && threadIdx.x == 0)
				{
					new_OW_displacement_ind = prev_OW_displacement_ind + (0.5*(prev_OW_speed_ind + new_OW_speed_ind)*curr_time_step_f);
					new_OW_speed_ind = prev_OW_speed_ind + (0.5*(prev_OW_acceleration_ind + new_OW_acceleration_ind)*curr_time_step_f);
				}

				new_TM_disp = prev_TM_disp_ind + (0.5*(prev_TM_sp_ind + new_TM_sp)*curr_time_step_f);

				// OHC: 

				new_ohc_psi_ind = prev_ohc_psi_ind + (0.5*(prev_ohcp_deriv_ind + new_ohcp_deriv_ind)*curr_time_step_f);

			}




			//_p_ohc = _model._gamma * _model._S_ohc * ( _TM_disp - _BM_disp );
			float ohc_disp = new_TM_disp - new_disp_vec[threadIdx.x]; //new_disp_ind;
			new_ohc_pressure_ind = gamma_ind * S_ohc_ind*ohc_disp;




			// Calc_DeltaL_OHC:
			// 
			// if (true == _model._OHC_NL_flag)
			//	_deltaL_disp = -_model._alpha_s * tanh( _model._alpha_L/_model._alpha_s * _psi_ohc );
			// else
			//	_deltaL_disp = -_model._alpha_L * _psi_ohc;

			float deltaL_disp;
			if (out_loop<CUDA_AUX_TM_LOOPS)
			{
				float aux_TM;
				if (enable_psi) // Non Linear PSI model
				{
					float tan_arg = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;


					tan_arg = tanhf(tan_arg);
					deltaL_disp = (-1.0 * _ohc_alpha_s)* tan_arg;

					//Calc_TM_speed
					//aux_TM = tanh(_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
					aux_TM = tan_arg*tan_arg;

				}
				else
				{
					deltaL_disp = -_ohc_alpha_l * new_ohc_psi_ind;

					//Calc_TM_speed
					//aux_TM = (_model._alpha_L/_model._alpha_s*_psi_ohc)^2;
					aux_TM = (__fdividef(_ohc_alpha_l, _ohc_alpha_s)) * new_ohc_psi_ind;
					aux_TM = aux_TM*aux_TM;

				}

				//Calc_TM_speed	 	
				// aux_TM = _model._alpha_L*( aux_TM - 1.0 );
				aux_TM = _ohc_alpha_l*(aux_TM - 1.0);




				// Numerator:
				//N_TM_sp = _p_ohc + _model._S_tm*_TM_disp - _model._gamma*( _model._R_tm*aux_TM*
				//			(_model._eta_2*_BM_sp + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc )
				//			+ _model._S_tm*_deltaL_disp );

				float N11;
				float N22;

				//float N_TM_sp = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				//N_TM_sp = N_TM_sp*R_tm_p*aux_TM + S_tm_p*deltaL_disp;
				//N_TM_sp = new_ohc_pressure_ind + S_tm_p*new_TM_disp - gamma_ind*N_TM_sp;


				N11 = eta_2*new_speed_vec[threadIdx.x]/*ind*/ + eta_1*ohc_disp - w_ohc*new_ohc_psi_ind;
				N22 = N11*R_tm_ind*aux_TM + S_tm_ind*deltaL_disp;
				float N_TM_sp = new_ohc_pressure_ind + S_tm_ind*new_TM_disp - gamma_ind*N22;

				// Denominator:
				//D_TM_sp = _model._R_tm*( _model._gamma*aux_TM*_model._eta_2 - 1.0 );

				float D_TM_sp = R_tm_ind*(gamma_ind*aux_TM*eta_2 - 1.0);

				new_TM_sp = __fdividef(N_TM_sp, D_TM_sp);

				// Calc_Ptm
				//_p_TM = _model._R_tm*_TM_sp + _model._S_tm*_TM_disp;

				P_TM_vec[threadIdx.x] = R_tm_ind*new_TM_sp + S_tm_ind*new_TM_disp;



			}
			// Calc_G   
			//_G = -1.0*( _p_TM + R_bm_nl()*_BM_sp + _model._S_bm*_BM_disp ); 

			// calc R_bm_nl

			float R_bm_nl;
			if (alpha_r == 0.0)
				R_bm_nl = R_ind;
			else
				R_bm_nl = (R_ind * (1.0 + (alpha_r * (new_speed_vec[threadIdx.x]/*ind*/ * new_speed_vec[threadIdx.x]/*ind*/))));



			float G_ind = -1.0*(P_TM_vec[threadIdx.x] + (R_bm_nl*new_speed_vec[threadIdx.x]/*ind*/) + (S_vec[threadIdx.x] * new_disp_vec[threadIdx.x]/*ind*/));


			float dx_pow2 = delta_x * delta_x;



			// Calc BC	& Calc_Y   		
			// _Y						= dx_pow2 * _G * _model._Q;	 
			// _Y[0]					= _bc;	 
			// _Y[_model._sections-1]	= 0.0;

			float new_sample;
			float _bc;

			if (threadIdx.x == 0)
			{
				//long nearest = long(__fdividef(curr_time_f - float_base_time_f, Ts));
				// relative position of x in the interval
				//float delta = __fdividef(((curr_time_f - float_base_time_f) - Ts * nearest), Ts);
				float delta = __fdividef(curr_time_f, Ts);


				new_sample = (sx2  * delta + sx1  * (1.0 - delta));




				if (enable_OW)
				{
					_bc = delta_x * model_a0 * (model_Gme * new_sample - model_a1 * new_OW_displacement_ind - model_a2 * new_OW_speed_ind);
				}
				else
				{
					_bc = delta_x * model_a0 * model_Gme *new_sample;
				}
				Y_ind = _bc;
			}
			else if (threadIdx.x == (blockDim.x - 1))
			{
				Y_ind = 0.0;
			}
			else
			{
				Y_ind = dx_pow2 * G_ind * Q_ind; //Q_vec[threadIdx.x];
			}

			// float _bc = new_sample; 



			// Jacobby -> Y
			int j_iters = (out_loop<2) ? _CUDA_JACOBBY_LOOPS1 : _CUDA_JACOBBY_LOOPS2;
			for (int iter = 0; (iter<j_iters/*_CUDA_JACOBBY_LOOPS*/); iter++)
			{
				__syncthreads();
				float tmp_p;
				tmp_p = (Y_ind - (l_vec[threadIdx.x + 1] * pressure_vec[threadIdx.x] + u_vec[threadIdx.x + 1] * pressure_vec[threadIdx.x + 1 + 1]))*m_inv; // /m_vec[threadIdx.x+1]; 	 		 


				__syncthreads();




				pressure_vec[threadIdx.x + 1] = tmp_p;


				// __threadfence();
			}



			// Calc_BM_acceleration	  
			//_BM_acc = ( _pressure + _G ) / _model._M_bm; 

			new_accel_ind = __fdividef(pressure_vec[threadIdx.x + 1] + G_ind, mass_ind);


			//__syncthreads(); 


			if (enable_OW && (threadIdx.x == 0))
			{

				//Calc_OW_Acceleration
				//_OW_acc = 1.0/_model._sigma_ow * ( _pressure[0] + _model._Gme * get_sample()
				//		  - _model._a2 * _OW_sp - _model._a1 * _OW_disp );


				new_OW_acceleration_ind = __fdividef(1.0, sigma_ow) *(pressure_vec[0 + 1] + model_Gme * new_sample - model_a2 * new_OW_speed_ind - model_a1 * new_OW_displacement_ind);


			}


			// Calc_Ohc_Psi_deriv
			//_d_psi_ohc = _model._eta_2*OHC_Speed() + _model._eta_1*OHC_Displacement() - _model._w_ohc*_psi_ohc;


			new_ohcp_deriv_ind = ((eta_1 * ohc_disp) + (eta_2 * (new_TM_sp - new_speed_vec[threadIdx.x]/*ind*/))) - (w_ohc * new_ohc_psi_ind);

			__syncthreads();

			//if (threadIdx.x<3)
			//{
			//	printf("threadIdx.x=%d ind=%d (outloop %d) rnl=%e G=%e Y=%e P=%e accel=%e ohcp_d=%e speed=%e\n",threadIdx.x,sample_loop,out_loop,rnl,G_ind,Y_ind,pressure_vec[threadIdx.x+1],new_accel_vec[threadIdx.x],new_ohcp_deriv_vec[threadIdx.x],new_speed_vec[threadIdx.x]);
			//} 


			//////////////////////////// TESTER //////////////////////////////



			//float prev_time_step = curr_time_step_f;

			// Lipschitz condition number
			//float	Lipschitz;			

			// find the speed error (compared to the past value)

			if (Lipschits_en && reset_iteration == 0) // if reset_iteration (first time) - do another loop
			{

				float sp_err = fabs(new_speed_vec[threadIdx.x]/*ind*/ - new_speed_ref);
				acc_diff_vec[threadIdx.x] = fabs(new_accel_ref - new_accel_ind);
				sp_err_vec[threadIdx.x] = sp_err;
				float sp_err_limit = fmax((float)(MAX_TOLERANCE), (float)(new_speed_vec[threadIdx.x] * curr_time_step_f));
				fault_vec[threadIdx.x] = (sp_err>sp_err_limit) ? 1 : 0;

				// TODO - take only sample into account?
				for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1)
				{
					__syncthreads();
					if (threadIdx.x<t_i)
					{
						sp_err_vec[threadIdx.x] = sp_err_vec[threadIdx.x] + sp_err_vec[threadIdx.x + t_i];
						fault_vec[threadIdx.x] = fault_vec[threadIdx.x] + fault_vec[threadIdx.x + t_i];
						acc_diff_vec[threadIdx.x] = fmax(acc_diff_vec[threadIdx.x], acc_diff_vec[threadIdx.x + t_i]);
					}

				}

				__syncthreads();


				// calculate lipschitz number 
				if (threadIdx.x<32)
				{
					float m1_sp_err = sp_err_vec[0];
					float Lipschitz;
					// Lipschitz calculations (threads 0-32)
					if (m1_sp_err > MAX_M1_SP_ERROR)
					{
						Lipschitz = __fdividef(acc_diff_vec[0], m1_sp_err);

					}
					else
						Lipschitz = 0;

					float Lmul = Lipschitz * curr_time_step_f;

					//int byp_cond = ((Lmul<=0.25) && (fault_vec[0]<=3)) ? 1 : 0;

					// use the calculated values to decide


					if (Lmul > 6.0) //2.0 )
					{
						// iteration didn't pass, step size decreased
						curr_time_step_f *= 0.5;
						if (curr_time_step_f<cuda_min_time_step)
						{
							curr_time_step_f = cuda_min_time_step;
						}

						another_loop = 0;
						reset_iteration = 1;
					} // Lmul >2
					else if (fault_vec[0] > 0)
					{
						// another iteration is needed for this step 
						another_loop = 1;
						reset_iteration = 0;
					} // faults > 0
					else if (Lmul < (float)(0.5))//(float)(0.25) )
					{
						// iteration passed, step size increased 
						curr_time_step_f *= 2.0;
						if (curr_time_step_f>cuda_max_time_step)
						{
							curr_time_step_f = cuda_max_time_step;
						}
						another_loop = 0;
						reset_iteration = 1;
					} // Lmul <0.25
					else
					{
						another_loop = 1;
						reset_iteration = 0;
					}
					sp_err_vec[0] = curr_time_step_f; // broadcast result
					sp_err_vec[1] = (float)another_loop;
					sp_err_vec[2] = (float)reset_iteration;

				} // threadIdx.x < 32



			} // !first_iteration
			else
			{

				if (threadIdx.x < 32)
				{
					sp_err_vec[0] = curr_time_step_f;
					sp_err_vec[1] = 1.0; // another loop 
					sp_err_vec[2] = 0.0; // reset_iteration 
				}
				//another_loop = 1;	 
				// reset_iteration = 0;
			}



			__syncthreads();


			curr_time_step_f = sp_err_vec[0];
			another_loop = (int)(sp_err_vec[1]);
			reset_iteration = (int)(sp_err_vec[2]);

			/////////////////////////////////////////////////////////



			out_loop++;
			if (out_loop >= oloop_limit)
			{
				another_loop = 0;
				reset_iteration = 1;
			}

		} // end of outern loop
		another_loop = 1;
		//float tdiff = curr_time_f - common_args_vec[0];
		//tdiff = tdiff > 0 ? tdiff : -tdiff;

		// if overlap time passed or its transient write and I'm actually write it	
		// replaced curr_time_f >= common_args_vec[0]
		if (curr_time_f > time_step_out)
		{
			// TODO
			curr_time_f -= time_step_out;
			input_sample++;
			sx1 = sx2;
			sx2 = input_samples[input_sample + 1];
			if (input_sample >= out_sample) {

				//int t_ind = SECTIONS*(out_sample+time_offset) + threadIdx.x;
				// index calculated here as the out sample index in block + time_offset from the start of the block + overlap blocks if part of transient interval, but not if the block of overlap
				// out sample already includes + time_offset + (show_transient - transient_offset)*overlap_nodes_for_block
				int t_ind = SECTIONS*input_sample + threadIdx.x;


				saved_speeds[t_ind] = new_speed_vec[threadIdx.x]/*ind*/;

				__syncthreads();


				if (threadIdx.x == 0) {
					common_args_vec[0] = common_args_vec[0] + time_step_out; // shared memory
				}


				out_sample++;

				__syncthreads(); // NEW NEW 

			}
		} // if write data is needed




		  // copy new to past



		prev_disp_ind = new_disp_vec[threadIdx.x]; //ind;
		prev_speed_ind = new_speed_vec[threadIdx.x]/*ind*/;
		prev_accel_ind = new_accel_ind;
		prev_ohc_psi_ind = new_ohc_psi_ind;
		prev_ohcp_deriv_ind = new_ohcp_deriv_ind;
		prev_TM_sp_ind = new_TM_sp;
		prev_TM_disp_ind = new_TM_disp;


		if (enable_OW && threadIdx.x == 0)
		{
			prev_OW_displacement_ind = new_OW_displacement_ind;
			prev_OW_speed_ind = new_OW_speed_ind;
			prev_OW_acceleration_ind = new_OW_acceleration_ind;
		}
		curr_time_f += curr_time_step_f;


		__syncthreads();



	} // end of sample loop

	  // store data in global mem

	  // TODO - signal finished section
	  // wait for new section to be ready (jumps of ...)



	__syncthreads();
}