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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h> 
#include <device_functions.h> 
#include <helper_functions.h>
#include <device_double_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <boost/stacktrace.hpp>
typedef float2 Complex;
#include "const.h"
#include "cochlea_common.h"	 
#include "cochlea.cuh"
//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>
//#include <thrust\fill.h>
#ifdef CUDA_MEX_PROJECT
#include <mex.h> 
#endif
// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


#ifndef gpuAssert
#define gpuAssert( condition ) { \
if( (condition) != 0 ) { \
std::ostringstream oss(""); \
oss << boost::stacktrace::stacktrace(); \
printf( "\n FAILURE %s in %s, line %d condition: %d, condition name : %s\n%s\n", cudaGetErrorString(condition), __FILE__, __LINE__ ,condition,cudaGetErrorName(condition),oss.str().c_str()); \
throw std::runtime_error("GPU Failure aborts..."); }\
 }
#endif
struct cudaHolderGeneratedData {
	float *generated_model_max_m1_sp_tolerance;
	float *generated_model_throw_tolerance;
	float *generated_calculated_power_array;  // in case of loaded from input boundary calculation write max power in dBSPL
	int	  *generated_model_out_sample_index;
	int   *generated_model_end_sample_index;
	int generated_sections = 0;

	void allocateGenerated() {
		if (generated_sections == 0) {
			generated_sections = MAX_NUMBER_OF_BLOCKS;
			int blocks_pointers_size = generated_sections * sizeof(float);
			int blocks_pointers_size_int = generated_sections * sizeof(int);
			// generated array  for blocks thresholds
			gpuAssert(cudaMalloc((void **)&generated_calculated_power_array, blocks_pointers_size));
			gpuAssert(cudaMalloc((void **)&generated_model_max_m1_sp_tolerance, blocks_pointers_size));
			gpuAssert(cudaMalloc((void **)&generated_model_throw_tolerance, blocks_pointers_size));

			gpuAssert(cudaMalloc((void **)&generated_model_out_sample_index, blocks_pointers_size_int));
			gpuAssert(cudaMalloc((void **)&generated_model_end_sample_index, blocks_pointers_size_int));
		}
	}

	void releaseGenerated() {
		if (generated_sections > 0) {
			generated_sections = 0;
			// generated array  for blocks thresholds
			gpuAssert(cudaFree(generated_calculated_power_array));
			generated_calculated_power_array = NULL;
			gpuAssert(cudaFree(generated_model_max_m1_sp_tolerance));
			generated_model_max_m1_sp_tolerance = NULL;
			gpuAssert(cudaFree(generated_model_throw_tolerance));
			generated_model_throw_tolerance = NULL;
			gpuAssert(cudaFree(generated_model_end_sample_index));
			generated_model_end_sample_index = NULL;
			gpuAssert(cudaFree(generated_model_out_sample_index));
			generated_model_out_sample_index = NULL;
		}
	}





} cudaHolderGeneratedData;
struct cudaModelAihc {
	int aihc_loaded = 0;
	int is_loaded() { return aihc_loaded;  }
	void loadAihc(float *Aihc) {
		if (!is_loaded()) {
			gpuAssert(cudaMemcpyToSymbol(model_Aihc, Aihc, SECTIONS*LAMBDA_COUNT * sizeof(float),0,cudaMemcpyHostToDevice));
			aihc_loaded = 1;
		}
	}
	void enableLoadAihc() {
		aihc_loaded = 0;
	}
} cudaModelAihc;

struct cudaHolderData {
	int cochlear_parametrs_initialized = 0;
	int cochlea_sections;
	
	
	float *cuda_input_samples;
	float *cuda_saved_speeds;
	float *cuda_Rd;
	float *cuda_Sd;
	float *cuda_Qd;
	float *cuda_Yd;
	float *cuda_gammad;
	float *converge_speed;
	float *converge_speed_blocks;




	float *cuda_massd;
	float *cuda_Md;
	float *cuda_Ud;
	float *cuda_Ld;
	float *cuda_S_ohcd;
	float *cuda_S_tmd;
	float *cuda_R_tmd;
	int *time_filter_params;
	int last_saved_nodes_per_time_block_for_cuda;
	int *cuda_Failed_Converged_Time_Node;
	int *cuda_Failed_Converged_Blocks;
	float *cuda_Converged_Time_Node;
	float *cuda_Converged_Blocks;
	float *cuda_convergence_jacoby_loops_per_iteration;
	float *cuda_convergence_jacoby_loops_per_iteration_blocks;
	size_t numBlocks_data = 0;
	size_t inputBufferNodes_data = 0;
	size_t resultBufferNodes_data = 0;
	void allocateCochlearData(const int& Sections) {
		if (cochlear_parametrs_initialized == 0) {
			cochlea_sections = Sections;
			cochlear_parametrs_initialized = 1;
			int cochlear_allocated = cochlea_sections * sizeof(float);
			// cuda Rd,Sd,Qd,Yd
			gpuAssert(cudaMalloc((void **)&cuda_Rd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Sd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Qd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Yd, cochlear_allocated));

			// cuda S_ohcd,S_tmd,gammad,R_tmd
			gpuAssert(cudaMalloc((void **)&cuda_S_ohcd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_S_tmd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_gammad, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_R_tmd, cochlear_allocated));

			// cuda massd,Md,Ud,Ld
			gpuAssert(cudaMalloc((void **)&cuda_massd, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Md, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Ud, cochlear_allocated));
			gpuAssert(cudaMalloc((void **)&cuda_Ld, cochlear_allocated));
		}

	}
	void releaseCochlearData() {
		if (cochlear_parametrs_initialized == 1) {
			// cuda Rd,Sd,Qd,Yd
			gpuAssert(cudaFree(cuda_Rd));
			cuda_Rd = NULL;
			gpuAssert(cudaFree(cuda_Sd));
			cuda_Sd = NULL;
			gpuAssert(cudaFree(cuda_Qd));
			cuda_Qd = NULL;
			gpuAssert(cudaFree(cuda_Yd));
			cuda_Yd = NULL;
			// cuda S_ohcd,S_tmd,gammad,R_tmd
			gpuAssert(cudaFree(cuda_S_ohcd));
			cuda_S_ohcd = NULL;
			gpuAssert(cudaFree(cuda_S_tmd));
			cuda_S_tmd = NULL;
			gpuAssert(cudaFree(cuda_gammad));
			cuda_gammad = NULL;
			gpuAssert(cudaFree(cuda_R_tmd));
			cuda_R_tmd = NULL;

			// cuda massd,Md,Ud,Ld
			gpuAssert(cudaFree(cuda_massd));
			cuda_massd = NULL;
			gpuAssert(cudaFree(cuda_Md));
			cuda_Md = NULL;
			gpuAssert(cudaFree(cuda_Ud));
			cuda_Ud = NULL;
			gpuAssert(cudaFree(cuda_Ld));
			cuda_Ld = NULL;
			cochlear_parametrs_initialized = 0;
		}
	}

	void loadCochlearData(float *S_ohc, float *S_tm, float *gamma, float *R_tm, float *mass, float *M, float *U, float *L, float *R, float *S,float *Q ) {
		if (cochlear_parametrs_initialized == 1) {
			int cochlea_allocated = cochlea_sections * sizeof(float);
			gpuAssert(cudaMemcpy(cuda_S_ohcd, S_ohc, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_S_tmd, S_tm, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_gammad, gamma, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_R_tmd, R_tm, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_massd, mass, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Md, M, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Ud, U, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Ld, L, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Rd, R, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Sd, S, cochlea_allocated, cudaMemcpyHostToDevice));
			gpuAssert(cudaMemcpy(cuda_Qd, Q, cochlea_allocated, cudaMemcpyHostToDevice));
		}
	
	}

	
	int isInputMemorySufficent(size_t test_nodes) { return inputBufferNodes_data >= test_nodes; }
	int isInputMemoryAllocated() { return isInputMemorySufficent(1); }

	int isOutputMemorySufficent(size_t test_nodes) { return resultBufferNodes_data >= test_nodes; }
	int isOutputMemoryAllocated() { return isOutputMemorySufficent(1); }

	int isBlocksMemorySufficent(size_t test_nodes) { return numBlocks_data >= test_nodes; }
	int isBlocksMemoryAllocated() { return isBlocksMemorySufficent(1); }

	void allocateOHCIOData(const size_t& inputBufferNodes,const size_t& resultBufferNodes) {
		if (!isInputMemorySufficent(inputBufferNodes)) {
			releaseInputData();
			size_t tsizep = inputBufferNodes * sizeof(float);
			size_t isizep = inputBufferNodes * sizeof(int);
			inputBufferNodes_data = inputBufferNodes;
			// input samples and results
			gpuAssert(cudaMalloc((void **)&cuda_input_samples, tsizep));
			gpuAssert(cudaMalloc((void **)&cuda_Failed_Converged_Time_Node, isizep));
			gpuAssert(cudaMalloc((void **)&cuda_Converged_Time_Node, tsizep));
			gpuAssert(cudaMalloc((void **)&cuda_convergence_jacoby_loops_per_iteration, tsizep));
			gpuAssert(cudaMalloc((void **)&converge_speed, tsizep));
		}
		if (!isOutputMemorySufficent(resultBufferNodes)) {
			releaseOutputData();
			size_t ssizep = resultBufferNodes * sizeof(float);
			resultBufferNodes_data = resultBufferNodes;
			printf("allocated %d bytes for BM velocity\n", ssizep);
			gpuAssert(cudaMalloc((void **)&cuda_saved_speeds, ssizep));
		}
	}

	void releaseInputData() {
		if (isInputMemoryAllocated()) {
			inputBufferNodes_data = 0;
			gpuAssert(cudaFree(cuda_input_samples));
			cuda_input_samples = NULL;
			gpuAssert(cudaFree(cuda_Failed_Converged_Time_Node));
			gpuAssert(cudaFree(converge_speed));
			gpuAssert(cudaFree(cuda_Converged_Time_Node));
			gpuAssert(cudaFree(cuda_convergence_jacoby_loops_per_iteration));
			cuda_convergence_jacoby_loops_per_iteration = NULL;
			cuda_Failed_Converged_Time_Node = NULL;
			cuda_Converged_Time_Node = NULL;
			converge_speed = NULL;
		}
	}
	void releaseOutputData() {
		if (isOutputMemoryAllocated()) {
			resultBufferNodes_data = 0;
			gpuAssert(cudaFree(cuda_saved_speeds));
			cuda_saved_speeds = NULL;
		}
	}
	void releaseOHCIOData() {
		// input samples and results
		releaseInputData();
		releaseOutputData();
		
	}

	void allocateBlocksConverganceArray(size_t nodes) {
		if (!isBlocksMemorySufficent(nodes)) {
			releaseBlocksConverganceArray();
			numBlocks_data = nodes;
			gpuAssert(cudaMalloc((void **)&cuda_Failed_Converged_Blocks, numBlocks_data*sizeof(int)));
			gpuAssert(cudaMalloc((void **)&converge_speed_blocks, numBlocks_data * sizeof(float)));
			gpuAssert(cudaMalloc((void **)&cuda_Converged_Blocks, numBlocks_data * sizeof(float)));
			gpuAssert(cudaMalloc((void **)&cuda_convergence_jacoby_loops_per_iteration_blocks, numBlocks_data * sizeof(float)));
			//converge_speed = thr
		}
	}
	void releaseBlocksConverganceArray() {
		if (isBlocksMemoryAllocated()) {
			gpuAssert(cudaFree(cuda_Failed_Converged_Blocks));
			cuda_Failed_Converged_Blocks = NULL;
			gpuAssert(cudaFree(converge_speed_blocks));
			gpuAssert(cudaFree(cuda_Converged_Blocks));
			gpuAssert(cudaFree(cuda_convergence_jacoby_loops_per_iteration_blocks));
			cuda_Converged_Blocks = NULL;
			converge_speed_blocks = NULL;
			cuda_convergence_jacoby_loops_per_iteration_blocks = NULL;
			numBlocks_data = 0;
		}
	}

} cudaHolderData;


struct cudaLambdaHolderData {
	// cuda IHC constant size data
	double *cuda_nIHC = NULL;
	int cochlea_sections = SECTIONS;
	int allocatedIHCDataVar = 0;
	inline int isIHCDataAllocated() { return allocatedIHCDataVar; }
	void allocateIHCData();
	void releaseIHCData();

	size_t allocatedLambdaNodes = 0; // to know if data need, allocated or reallocated
	inline size_t isLambdaMemorySufficent(size_t test_nodes) { return allocatedLambdaNodes >= LAMBDA_COUNT*test_nodes; }
	inline size_t isLambdaMemoryAllocated() { return isLambdaMemorySufficent(1); }

	size_t allocatedBufferNodes = 0; // to know if data need, allocated or reallocated
	inline int isBufferMemorySufficent(size_t test_nodes) { return allocatedBufferNodes >= test_nodes; }
	inline int isBufferMemoryAllocated() { return isBufferMemorySufficent(1); }

	JNDFloat *cuda_JND_Lambda = NULL; // will use as buffer too for memory conversions purposes
	JNDFloat *cuda_Buffer1 = NULL;



	// nodes count for single buffer, for full lambda count multiple by LAMBDA constant
	void allocateLambdaMemory(size_t nodes);

	void releaseLambdaMemory();

	// nodes count for single buffer, for full lambda count multiple by LAMBDA constant
	void allocateBufferMemory(size_t nodes);

	void releaseBufferMemory();

} cudaLambdaHolderData;


void  cudaLambdaHolderData::releaseIHCData() {
	if (isIHCDataAllocated()) {

		gpuAssert(cudaFree(cuda_nIHC));
		cuda_nIHC = NULL;
		allocatedIHCDataVar = 0;
	}
}

void cudaLambdaHolderData::allocateIHCData() {
	if (!isIHCDataAllocated()) {
		gpuAssert(cudaMalloc((void **)&cuda_nIHC, cochlea_sections * sizeof(double)));
		allocatedIHCDataVar = 1;
	}
}


// nodes count for single buffer, for full lambda count multiple by LAMBDA constant
void cudaLambdaHolderData::allocateLambdaMemory(size_t nodes) {
	if (!isLambdaMemorySufficent(nodes)) {
		releaseLambdaMemory();
		allocatedLambdaNodes = nodes*LAMBDA_COUNT; // now its sufficent
		size_t lambda_memory_size_in_bytes = allocatedLambdaNodes * sizeof(JNDFloat);
		printf("allocated %lu bytes for lambda nodes\n", lambda_memory_size_in_bytes);
		gpuAssert(cudaMalloc((void **)&cuda_JND_Lambda, lambda_memory_size_in_bytes));
	}
}

void cudaLambdaHolderData::releaseLambdaMemory() {
	if (isLambdaMemoryAllocated()) {
		gpuAssert(cudaFree(cuda_JND_Lambda));
		cuda_JND_Lambda = NULL;
		allocatedLambdaNodes = 0;
	}
}

// nodes count for single buffer, for full lambda count multiple by LAMBDA constant
void cudaLambdaHolderData::allocateBufferMemory(size_t nodes) {
	if (!isBufferMemorySufficent(nodes)) {
		releaseBufferMemory();
		allocatedBufferNodes = nodes; // now its sufficent
		size_t lambda_memory_size_in_bytes = allocatedBufferNodes * sizeof(JNDFloat);
		printf("allocated %lu bytes for buffer nodes\n", lambda_memory_size_in_bytes);
		gpuAssert(cudaMalloc((void **)&cuda_Buffer1, lambda_memory_size_in_bytes));
	}
}

void cudaLambdaHolderData::releaseBufferMemory() {
	if (isBufferMemoryAllocated()) {
		gpuAssert(cudaFree(cuda_Buffer1));
		cuda_Buffer1 = NULL;
		allocatedBufferNodes = 0;
	}
}

extern "C" JNDFloat *getCudaBuffer() {

	return cudaLambdaHolderData.cuda_Buffer1;
}

extern "C" JNDFloat *getCudaLambda() {

	return cudaLambdaHolderData.cuda_JND_Lambda;
}

extern "C" int *getCudaFailedTimeNodes() {
	return cudaHolderData.cuda_Failed_Converged_Time_Node;
}

extern "C" int *getCudaFailedBlocks() {
	return cudaHolderData.cuda_Failed_Converged_Blocks;
}


extern "C" float *getCudaConvergedTimeNodes() {
	return cudaHolderData.cuda_Converged_Time_Node;
}

extern "C" float *getCudaConvergedJacobyLoopsPerIteration() {
	return cudaHolderData.cuda_convergence_jacoby_loops_per_iteration;
}

extern "C" float *getCudaConvergedJacobyLoopsPerIterationBlocks() {
	return cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks;
}

extern "C" float *getCudaConvergedBlocks() {
	return cudaHolderData.cuda_Converged_Blocks;
}

extern "C" void loadAihc(float *Aihc) noexcept(false) {
	cudaModelAihc.loadAihc(Aihc);
}

extern "C" void enableloadAihc() noexcept(false) {
	cudaModelAihc.enableLoadAihc();
}
extern "C" void extractConvergenceTimes(float *convergence_times, size_t nodes) {
	gpuAssert(cudaMemcpy(convergence_times, cudaHolderData.converge_speed, nodes * sizeof(float), cudaMemcpyDeviceToHost));
}
int *host_params_time_filter,*host_time_filters;

struct cudaJNDHolder {
	//JNDFloat *cuda_dLambda = NULL;


	JNDFloat *cuda_MeanRate = NULL;


	/**
	* pointer to parameters array of structures on the host
	*/
	device_params *host_local_param;
	/**
	* pointer to parameters array of structures on global device memory
	*/
	device_params *global_device_params;
	vectors_sum_linear_coefficents *vectors_sums_coefficents;

	int cuda_device_jnd_structs_allocated = 0;
	int isDeviceStructsAllocated() { return cuda_device_jnd_structs_allocated; }
	void ReleaseDeviceStructs() {
		if (isDeviceStructsAllocated()) {
			cuda_device_jnd_structs_allocated = 0;
			delete[](host_local_param);
			gpuAssert(cudaFree(global_device_params));
			global_device_params = NULL;
			gpuAssert(cudaFree(vectors_sums_coefficents));
			vectors_sums_coefficents = NULL;
		}
	}

	void allocateDeviceStructs() {
		if (!isDeviceStructsAllocated()) {
			gpuAssert(cudaMalloc((void **)&vectors_sums_coefficents, 2 * sizeof(vectors_sum_linear_coefficents)));
			gpuAssert(cudaMalloc((void **)&global_device_params, 2 * sizeof(device_params))); // just one set filter includes its own size
			host_local_param = new device_params[2];
			cuda_device_jnd_structs_allocated = 1;
		}
	}


	size_t cuda_jnd_intervals_num = 0;
	int isSufficentIntervalAllocated(size_t nodes) { return cuda_jnd_intervals_num >= nodes; }
	int isIntervalsAllocated() { return isSufficentIntervalAllocated(1); }

	device_jnd_params *cuda_jnd_params = NULL;
	int *cuda_JND_Serial_Intervals_Positions = NULL;
	int *cuda_JND_Interval_To_Reference = NULL;
	int *cuda_JND_Calculated_Intervals = NULL;
	int *cuda_JND_Refrence_Intervals = NULL;

	void releaseIntervals() {
		if (isIntervalsAllocated()) {
			printf("clearing %d params\n", cuda_jnd_intervals_num);
			gpuAssert(cudaFree(cuda_JND_Serial_Intervals_Positions));
			gpuAssert(cudaFree(cuda_JND_Interval_To_Reference));
			gpuAssert(cudaFree(cuda_JND_Calculated_Intervals));
			gpuAssert(cudaFree(cuda_JND_Refrence_Intervals));
			gpuAssert(cudaFree(cuda_jnd_params));
			cuda_JND_Serial_Intervals_Positions = NULL;
			cuda_JND_Interval_To_Reference = NULL;
			cuda_JND_Calculated_Intervals = NULL;
			cuda_JND_Refrence_Intervals = NULL;
			cuda_jnd_params = NULL;
			cuda_jnd_intervals_num = 0;
		}
	}

	void allocateIntervals(int intervals_num) {
		if (!isSufficentIntervalAllocated(intervals_num)) {
			releaseIntervals();

			cuda_jnd_intervals_num = intervals_num;
			size_t jndRefrencesSizeInBytes = cuda_jnd_intervals_num * sizeof(device_jnd_params);

			size_t dA_size_in_bytes = cuda_jnd_intervals_num * sizeof(device_jnd_params);
			gpuAssert(cudaMalloc((void **)&cuda_jnd_params, dA_size_in_bytes));
			gpuAssert(cudaMalloc((void **)&cuda_JND_Serial_Intervals_Positions, jndRefrencesSizeInBytes));
			gpuAssert(cudaMalloc((void **)&cuda_JND_Interval_To_Reference, jndRefrencesSizeInBytes));
			gpuAssert(cudaMalloc((void **)&cuda_JND_Calculated_Intervals, jndRefrencesSizeInBytes));
			gpuAssert(cudaMalloc((void **)&cuda_JND_Refrence_Intervals, jndRefrencesSizeInBytes));
		}
	}

	JNDFloat *cuda_FisherAISum = NULL;
	JNDFloat *cuda_F_RA = NULL;

	size_t cuda_fisher_size = 0;
	int isSufficentFisherNodesAllocated(size_t nodes) { return cuda_fisher_size >= nodes; }
	int isFisherNodesAllocated() { return isSufficentFisherNodesAllocated(1); }
	void allocateFisherNodes(size_t nodes);
	void releaseFisherNodes();

	size_t cuda_mean_nodes = 0;

	int isSufficentMeanNodesAllocated(size_t nodes) { return cuda_mean_nodes >= nodes; }
	int isMeanNodesAllocated() { return isSufficentMeanNodesAllocated(1); }
	void allocateMeanNodes(size_t nodes);
	void releaseMeanNodes();

} cudaJNDHolder;

void cudaJNDHolder::releaseFisherNodes() {

	if (isFisherNodesAllocated()) {
		printf("clearing %d fisher nodes\n", cuda_fisher_size);
		gpuAssert(cudaFree(cuda_FisherAISum));
		gpuAssert(cudaFree(cuda_F_RA));
		cuda_FisherAISum = NULL;
		cuda_F_RA = NULL;
		cuda_fisher_size = 0;
	}
}

void cudaJNDHolder::allocateFisherNodes(size_t nodes) {
	if (!isSufficentFisherNodesAllocated(nodes)) {
		releaseFisherNodes();
		cuda_fisher_size = nodes;
		size_t fisher_size_in_bytes = cuda_fisher_size * sizeof(JNDFloat);
		gpuAssert(cudaMalloc((void **)&cuda_FisherAISum, fisher_size_in_bytes));
		//gpuAssert(cudaMalloc((void **)&cuda_AvgMeanRate, fisher_size_in_bytes));
		gpuAssert(cudaMalloc((void **)&cuda_F_RA, fisher_size_in_bytes));
	}
}

void cudaJNDHolder::releaseMeanNodes() {

	if (isMeanNodesAllocated()) {
		printf("clearing %d mean nodes\n", cuda_mean_nodes);
		gpuAssert(cudaFree(cuda_MeanRate));
		cuda_MeanRate = NULL;
		cuda_mean_nodes = 0;
	}
}

void cudaJNDHolder::allocateMeanNodes(size_t nodes) {
	if (!isSufficentMeanNodesAllocated(nodes)) {
		releaseMeanNodes();
		cuda_mean_nodes = nodes; 
		size_t mean_size_in_bytes = cuda_mean_nodes * sizeof(JNDFloat);
		gpuAssert(cudaMalloc((void **)&cuda_MeanRate, mean_size_in_bytes));
	}
}


extern "C" JNDFloat *getCudaMeanRate() {

	return cudaJNDHolder.cuda_MeanRate;
}


struct cudaSignalHolder {
	size_t cuda_signal_nodes = 0;
	float *cuda_WN = NULL; // white noise for input generation
	float *cuda_Signal = NULL;
	int isSufficentSignalNodesAllocated(size_t nodes) { return cuda_signal_nodes >= nodes; }
	int isSignalNodesAllocated() { return isSufficentSignalNodesAllocated(1); }
	void allocateSignalNodes(int nodes) {
		if (!isSufficentSignalNodesAllocated(nodes)) {
			releaseSignalNodes();
			cuda_signal_nodes = nodes;
			size_t wn_length_bytes = cuda_signal_nodes * sizeof(float);
			gpuAssert(cudaMalloc((void **)&cuda_WN, wn_length_bytes));
			gpuAssert(cudaMalloc((void **)&cuda_Signal, wn_length_bytes));
		}
	}
	void releaseSignalNodes() {
		if (isSignalNodesAllocated()) {
			cuda_signal_nodes = 0;
		}
	}
} cudaSignalHolder;
/****
* cochlea cu global variables
*
*
**/


	
	//float *deviceBackupSpeeds; // save backup speeds from previous ihc run on the device
	float *BM_host;

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
   cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
		printf("cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		/**
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );*/

		throw std::runtime_error("Cuda check pre device synchronize failed");
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
		/**
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
		*/
		printf("cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));

		throw std::runtime_error("Cuda check post device synchronize failed");
    }
#endif
 
    return;
}


inline void __cudaClearError( const char *file, const int line )
{
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
   cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
		/**
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );*/

		printf("cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
    }
 
    return;
}



std::string showDIM3(dim3 d3) {
	std::stringstream	ss;
	ss << "(" << d3.x << ", " << d3.y << "," << d3.z << ")";
	return ss.str();
}

extern "C" void cudaEventsCreate(cudaEvent_t& start, cudaEvent_t& stop, int condition) noexcept(false) {
	if (condition) {
		gpuAssert(cudaEventCreate(&start));
		gpuAssert(cudaEventCreate(&stop));
	}
}
extern "C" void cudaEventsStartTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition) noexcept(false) {
	if (condition) {
		gpuAssert(cudaEventRecord(start));
	}
}
extern "C" void viewGPUStatus(int flags, const std::string& prefix) noexcept(false) {
	if (flags & 16) {
		size_t free_memory;
		size_t total_memory;
		gpuAssert(cudaMemGetInfo(&free_memory, &total_memory));
		printf("%s : GPU Memory, Free(%d MB) / Total(%d MB)\n",prefix.c_str(), static_cast<int>((free_memory / (1024 * 1024))), static_cast<int>(total_memory / (1024 * 1024)));
	}
}
extern "C" void cudaEventsStopTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition) noexcept(false) {
	if (condition) {
		gpuAssert(cudaEventRecord(stop));
	}
}
extern "C" float *getCudaConvergeSpeedBlocks() {
	return cudaHolderData.converge_speed_blocks;
}

extern "C" float cudaEventsStopQueryTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition, const std::string& prefix) noexcept(false) {
	cudaEventsStopTimer(start, stop, condition);
	float milliseconds = 0.0f;
	if (condition) {
		gpuAssert(cudaEventSynchronize(stop));
		
		gpuAssert(cudaEventElapsedTime(&milliseconds, start, stop));
		printf("%s : %.2f (msec) \n", prefix.c_str(), milliseconds);
	}
	return milliseconds;
}
template void GeneralKernel_Copy_Results_Template<double>(double *target, double *src, size_t size);
template void GeneralKernel_Copy_Results_Template<float>(float *target, float *src, size_t size);
template void GeneralKernel_Copy_Results_Template<int>(int *target, int *src, size_t size);

template void GeneralKernel_Copy_Results_Template<double>(double *target, double *src, size_t size, size_t offset);
template void GeneralKernel_Copy_Results_Template<float>(float *target, float *src, size_t size, size_t offset);
template void GeneralKernel_Copy_Results_Template<int>(int *target, int *src, size_t size, size_t offset);


template void ReverseKernel_Copy_Results_Template<float>(float *cpu_src, float *cuda_target, size_t start_time_node, size_t time_nodes, int sections);
template void ReverseKernel_Copy_Results_Template<double>(double *cpu_src, double *cuda_target, size_t start_time_node, size_t time_nodes, int sections);

template void updateCUDALambdaArray<float>(float *lambda_array, float *cuda_buffer, size_t calc_time_nodes, int sections, int Show_Run_Time, int Show_Device_Data, int cuda_buffer_update, Log &outer_log);
template void updateCUDALambdaArray<double>(double *lambda_array,double *cuda_buffer, size_t calc_time_nodes, int sections, int Show_Run_Time, int Show_Device_Data, int cuda_buffer_update,Log &outer_log);


extern "C" void BMOHCKernel_Init( 
float *gamma, 
float *mass,
float *M,
float *U,
float *L,
float *R,
float *S,
float *Q,
float *S_ohc,
float *S_tm,
float *R_tm,
float num_frequencies,
float dbA,
size_t inputBufferNodes,
size_t resultBufferNodes,
size_t lambdaBufferNodes,
bool first_time,
int Show_Run_Time,
int Show_Device_Data,
Log &outer_log
){




   cudaEvent_t start, stop;

   size_t ssizep = resultBufferNodes*sizeof(float);
  
   cudaEventsCreate(start, stop, Show_Run_Time & 2);
   cudaEventsStartTimer(start, stop, Show_Run_Time & 2);
   if ( first_time ) {
	
	if (Show_Device_Data & 8) {
		printf("Saved speeeds allocated size = (%d KB), %d Nodes\n", (ssizep / 1024), (ssizep / 256));
	}
	

	
   } // end of first time memory allocations

   cudaHolderGeneratedData.allocateGenerated();
   cudaHolderData.allocateOHCIOData(inputBufferNodes, resultBufferNodes);
	//printf("memory uploads lambda program arrays\n");
   cudaHolderData.allocateCochlearData(SECTIONS);
   cudaHolderData.loadCochlearData(S_ohc, S_tm, gamma, R_tm, mass, M, U, L, R, S, Q);
	//printf("cuda malloc fisher program arrays\n");
	outer_log.timeAtFlag(0,cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 2,"Initialize and allocate Memory for BM run"), Show_Run_Time & 2);
		
   
}




extern "C" void BMOHCKernel_Wait_Threads() noexcept(false)
{
	gpuAssert(cudaDeviceSynchronize());
}
extern "C" void BMOHCKernel_Copy_Results(float *target, size_t resultNodes, size_t offset) noexcept(false) {

	size_t ssize = resultNodes*sizeof(float);
	gpuAssert(cudaMemcpy((void *)(target), cudaHolderData.cuda_saved_speeds+offset, ssize, cudaMemcpyDeviceToHost)); 
}

extern "C" void ReverseKernel_Copy_Results(float *src, size_t size) noexcept(false) {
	gpuAssert(cudaMemcpy((void *)cudaHolderData.cuda_saved_speeds,src, size*sizeof(float), cudaMemcpyHostToDevice)); 
}


extern "C" void BMOHCKernel_Copy_Lambda(JNDFloat *target, size_t lambdaNodes, int offset) noexcept(false) {
	size_t lsizep = lambdaNodes*sizeof(JNDFloat);
	gpuAssert(cudaMemcpy((void *)(target), cudaLambdaHolderData.cuda_Buffer1+offset, lsizep, cudaMemcpyDeviceToHost));
}
   
extern "C" void BMOHCKernel_Free(
) noexcept(false) {
	cudaHolderData.releaseCochlearData();
	cudaHolderGeneratedData.releaseGenerated();
	cudaHolderData.releaseOHCIOData();
	cudaHolderData.releaseBlocksConverganceArray();
}
// non adjusted version for relative error parmeters as 0
__global__ void CudaCalculateThresholdBoundariesForNonGeneratedInputSimple(
	float *m1_sp_maximum,
	float *tolerance_maximum
	) {
	m1_sp_maximum[threadIdx.x] = model_constants[23];
	tolerance_maximum[threadIdx.x] = model_constants[24];
	__syncthreads();
}

// calculate threshold boundaries based on read input from file
__global__ void CudaCalculateThresholdBoundariesForNonGeneratedInput(
	float *input_samples,
	float *m1_sp_maximum,
	float *tolerance_maximum,
	float *power_calculated_array
	) {
	__shared__ float blockMaximum[1024];
	//__shared__ float loader[1024];
	int start_input = model_constants_integers[8] * blockIdx.x;
	int bdim = blockDim.x;
	float load_data = input_samples[start_input + threadIdx.x];
	load_data = abs(load_data);
	blockMaximum[threadIdx.x] = load_data;
	
	for (int t_i = (bdim >> 1); t_i >= 1; t_i >>= 1) {
		__syncthreads();
		if (threadIdx.x < t_i ) {
			blockMaximum[threadIdx.x] = fmax(blockMaximum[threadIdx.x], blockMaximum[threadIdx.x + t_i]);
		}

	}
	__syncthreads();
	if (threadIdx.x ==0) {
		// calculate thresholds
		// first calculate power relative to SPLRef
		float power_calculated = model_constants[34] * blockMaximum[0];
		power_calculated_array[blockIdx.x] = power_calculated;
	}
	__syncthreads();
}


// calculate threshold boundaries based on read input from file
__global__ void CudaCalculateThresholdBoundariesForNonGeneratedInputBlocked(
	volatile float *input_samples,
	volatile float *power_calculated_array,
	int block_size,
	float rSPLRefVal,
	float M1_SP_Fix_Factor,
	float Tolerance_Fix_Factor,
	float Max_M1_SP_Error_Parameter,
	float Max_Tolerance_Parameter,
	int *gen_model_end_sample_index
) {
	//__shared__ float loader[1024];
	int start_input = block_size * threadIdx.x;
	int end_output = gen_model_end_sample_index[threadIdx.x];
	//int end_output = (model_constants_integers[8]+1) * threadIdx.x;
	// determine max output for boundaries conditions
	float load_data = 0.0f;
	load_data = input_samples[start_input];
	for (int t_i = start_input; t_i <end_output; t_i++) {
		load_data = fmax(load_data, input_samples[t_i]);

	}
		// calculate thresholds
		// first calculate power relative to SPLRef
	//float power_calculated = rSPLRefVal * load_data;
	power_calculated_array[threadIdx.x] = rSPLRefVal * load_data;
	__syncthreads();
}

extern "C" void calculateBoundariesForNonGeneratedInput(
	int Relative_Error_Parameters,
	int max_block_length,
	int Show_Calculated_Power, 
	float M1_SP_Fix_Factor,
	float Tolerance_Fix_Factor,
	float Max_M1_SP_Error_Parameter,
	float Max_Tolerance_Parameter,
	float rSPLRefVal,
	int block_size,
	dim3 inputBlockDivision
	) noexcept(false) {
	dim3 singleton(1, 1, 1);

	if (Relative_Error_Parameters == 0) {
		printf("Calculating simple configuration for %d blocks\n", inputBlockDivision.x);
		CudaCalculateThresholdBoundariesForNonGeneratedInputSimple KERNEL_ARGS2(singleton, inputBlockDivision)(cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance, cudaHolderGeneratedData.generated_model_throw_tolerance);
		
	} else {
		float host_generated_m1_sp_array[MAX_NUMBER_OF_BLOCKS];
		float host_generated_tolerance_array[MAX_NUMBER_OF_BLOCKS];
		float host_generated_calculated_power_array[MAX_NUMBER_OF_BLOCKS];
		int threadsNumber = min(max_block_length, 1024);
		if (threadsNumber < 1024) {
			threadsNumber = static_cast<int>(powf(2.0f, floor(log2f(static_cast<float>(threadsNumber)))));
		}
		dim3 threadsDivision(threadsNumber, 1, 1);
		printf("Calculating alternate complex configuration size of %d\n", inputBlockDivision.x); // " blocks  divided to threads at " << showDIM3(threadsDivision) << std::endl;
		CudaCalculateThresholdBoundariesForNonGeneratedInput KERNEL_ARGS2(inputBlockDivision, threadsDivision)(cudaHolderData.cuda_input_samples, cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance, cudaHolderGeneratedData.generated_model_throw_tolerance, cudaHolderGeneratedData.generated_calculated_power_array);
		gpuAssert(cudaMemcpy(host_generated_calculated_power_array, cudaHolderGeneratedData.generated_calculated_power_array, static_cast<int>(inputBlockDivision.x)*sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < static_cast<int>(inputBlockDivision.x); i++) {
			float power_calculated = host_generated_calculated_power_array[i] > 0.0f ? 20 * log10f(0.1f*host_generated_calculated_power_array[i]) : 0.0f;
			host_generated_m1_sp_array[i] = Max_M1_SP_Error_Parameter*powf(10.0f, M1_SP_Fix_Factor * power_calculated);
			host_generated_tolerance_array[i] = Max_Tolerance_Parameter*powf(10.0f, Tolerance_Fix_Factor * power_calculated);
			if (Show_Calculated_Power&1) {

				printf("generated_calculated_power_array[%d]=%.3e\n",i, power_calculated);
				printf("generated_m1_sp_array[%d]=%.3e\n", i, host_generated_m1_sp_array[i]);
				printf("generated_tolerance_array[%d]=%.3e\n", i, host_generated_tolerance_array[i]);

			}

		}
	}
}
/////////////////////////////////////////////////

void cudaOccupancyIndicator(int blockSize, const void *MyKernel, cudaDeviceProp &deviceProp) {

	int maxActiveBlocks;
	gpuAssert(cudaDeviceSynchronize());
	gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
		MyKernel, blockSize,
		0));


	float occupancy = (maxActiveBlocks * blockSize / deviceProp.warpSize) /
		(float)(deviceProp.maxThreadsPerMultiProcessor /
			deviceProp.warpSize);

	printf("Launched blocks of size %d. Theoretical occupancy: %f,maxActive Blocks: %d\n",
		blockSize, occupancy, maxActiveBlocks);
	/*
	std::cout << "activeWarps: " << activeWarps << std::endl
		<< "maxWarps: " << maxWarps << std::endl
		<< "GPU Blocks processing capability of BM calculations (" << run_mode_name << ") is : " << (deviceProp.multiProcessorCount*numBlocks) << std::endl
		<< "current num blocks: " << grid.x << std::endl
		<< "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
		*/
}

extern "C" void BMOHCNewKernel(

float *input_samples,
bool override_input_samples, // true if input generated, will not upload
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
float *tres,
int Time_Blocks,
int samplesBufferLengthP1,
int overlap_nodes_for_block,
long overlapTimeMicroSec,
int show_transient, // always 1 and will be ignored than
float cuda_max_time_step,
float cuda_min_time_step,
int Decouple_Filter,
float Max_M1_SP_Error_Parameter,
float Max_Tolerance_Parameter,
int Relative_Error_Parameters,
int calculate_boundary_conditions, // if true will calculate max tolerance and max m1 sp error from input, should be used if input is not generated within the program	
float M1_SP_Fix_Factor,
float Tolerance_Fix_Factor,
float SPLREfVal,
int Show_Calculated_Power,
int Show_Device_Data,
int Show_Run_Time,
int JACOBBY_Loops_Fast, // number of jcoby loops to perform on fast approximation
int JACOBBY_Loops_Slow, // number of jcoby loops to perform on slow approximation
int Cuda_Outern_Loops, // max control loops
int Run_Fast_BM_Calculation, // will run BM calculation with relaxed memory requirements
int BMOHC_Kernel_Configuration,
cudaEvent_t& start, 
cudaEvent_t& stop,
cudaDeviceProp deviceProp,
Log &outer_log

) noexcept(false) {
	

	cudaEventsCreate(start, stop, Show_Run_Time & 1);
    dim3 threads(FIRST_STAGE_WIDTH, 1);
	dim3 grid(Time_Blocks/*FIRST_STAGE_BLOCKS*/, 1);
	cudaHolderData.allocateBlocksConverganceArray(Time_Blocks);
	//last_saved_nodes_per_time_block_for_cuda = last_saved_nodes_per_block; // this setup for later copy data, fix indexes
	int tsizep = (samplesBufferLengthP1)*sizeof(float);
	//std::cout << "inputing " << samplesBufferLengthP1 << " nodes\n"<<tsizep<<" Bytes\n";
	//  allocate memory on device
 
 
		
   // copy data to device
		
		// TODO - allocate & memcopy only neccesary data according to enable_ow and enable_psi
		

	//int numBlocks = grid.x; // Occupancy in terms of active blocks
	//int activeWarps; 
	//int maxWarps; 
	std::string run_mode_name;
	if (Run_Fast_BM_Calculation == 3 ) {
		run_mode_name = "fast no self analysis";
	} else if (Run_Fast_BM_Calculation == 2) {
		run_mode_name = "Impercise";
	} else if (Run_Fast_BM_Calculation == 1) {
		run_mode_name = "Fast";
	} else {
		run_mode_name = "Precise";
	}
	/*
	if (Show_Device_Data & 2) {
		
		if (Run_Fast_BM_Calculation) {
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, BMOHC_FAST_kernel, threads.x, 0);
		} else {
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, BMOHC_NEW_kernel, threads.x, 0);
		}
		activeWarps = deviceProp.multiProcessorCount* numBlocks * threads.x / deviceProp.warpSize;
		maxWarps = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
		std::cout << "activeWarps: " << activeWarps << std::endl
			<< "maxWarps: " << maxWarps << std::endl
			<< "GPU Blocks processing capability of BM calculations ("<<run_mode_name<<") is : " << (deviceProp.multiProcessorCount*numBlocks) << std::endl
			<< "current num blocks: " << grid.x << std::endl
			<< "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
	}
	*/
	
		
	if (!override_input_samples) {
		gpuAssert(cudaMemcpy(cudaHolderData.cuda_input_samples, input_samples, tsizep, cudaMemcpyHostToDevice));
	}
	// load GPU parmeters
	float host_model_constants[MODEL_FLOATS_CONSTANTS_SIZE];
	int host_model_constants_integers[MODEL_INTEGERS_CONSTANTS_SIZE];
	int host_model_constants_longs[MODEL_LONGS_CONSTANTS_SIZE];
	float base_time = 0;
	host_model_constants[0] = Ts;
	host_model_constants[1] = static_cast<float>(1.0/static_cast<double>(Ts)); // Fs
	host_model_constants[2] = _ohc_alpha_l;
	host_model_constants[3] = _ohc_alpha_s; 
	host_model_constants[4] = -1.0f*_ohc_alpha_l;
	host_model_constants[5] = -1.0f*_ohc_alpha_s;
	host_model_constants[6] = static_cast<float>(static_cast<double>(_ohc_alpha_l) / static_cast<double>(_ohc_alpha_s));
	host_model_constants[7] = static_cast<float>(1.0/static_cast<double>(sigma_ow));
	host_model_constants[8] = delta_x;
	host_model_constants[9] = delta_x*delta_x; //  dx_pow2
	host_model_constants[10] = model_a0;
	host_model_constants[11] = model_a1;
	host_model_constants[12] = model_a2;
	host_model_constants[13] = eta_1;
	host_model_constants[14] = eta_2;
	host_model_constants[15] = model_Gme;
	host_model_constants[16] = enable_OW*model_Gme;
	host_model_constants[17] = (1 - enable_OW)*model_Gme;
	host_model_constants[18] = -w_ohc; // negated, so all negatives in GPU become positive
	host_model_constants[19] = time_step;
	host_model_constants[20] = time_step_out;
	//std::cout << "host_model_constants[20] = time_step_out = " << time_step_out << std::endl;
	host_model_constants[21] = cuda_min_time_step;
	host_model_constants[22] = cuda_max_time_step;
	host_model_constants[23] = Max_M1_SP_Error_Parameter;
	host_model_constants[24] = Max_Tolerance_Parameter;
	//printf("model_constants[25] = alpha_r=%.2e\n", alpha_r);
	host_model_constants[25] = alpha_r;
	host_model_constants[26] = host_model_constants[8] * host_model_constants[10];
	host_model_constants[31] = -model_a1;
	host_model_constants[32] = -model_a2;
	host_model_constants[33] = SPLREfVal;
	host_model_constants[34] = 1.0f/SPLREfVal;
	host_model_constants[35] = M1_SP_Fix_Factor;
	host_model_constants[36] = Tolerance_Fix_Factor;

	// integer constants
	host_model_constants_integers[0] = Decouple_Filter;
	//std::cout << "Decouple_Filter = " << Decouple_Filter << std::endl;
	host_model_constants_integers[1] = enable_OW;
	host_model_constants_integers[2] = enable_psi;
	host_model_constants_integers[3] = Time_Blocks;
	host_model_constants_integers[4] = samplesBufferLengthP1;
	host_model_constants_integers[5] = overlap_nodes_for_block;
	host_model_constants_integers[6] = show_transient;
	host_model_constants_integers[7] = Relative_Error_Parameters;
	host_model_constants_integers[8] = (samplesBufferLengthP1 - overlap_nodes_for_block) / (Time_Blocks + 1);
	host_model_constants_integers[9] = JACOBBY_Loops_Fast;
	host_model_constants_integers[10] = JACOBBY_Loops_Slow;
	host_model_constants_integers[11] = Cuda_Outern_Loops;
	int host_model_out_sample_index[MAX_NUMBER_OF_BLOCKS];
	int host_model_end_sample_index[MAX_NUMBER_OF_BLOCKS];
	int max_block_length = 0;
	// calculate division of of timed blocks, which node each cuda block starts procerssing the input
	for (int bindex = 0; bindex < static_cast<int>(grid.x); bindex++) {
		int transient_offset =  (bindex > 0 && (host_model_constants_integers[0] == 0 || (host_model_constants_integers[0] != 1 && bindex%host_model_constants_integers[0] != 0)));
		int preDecoupled = host_model_constants_integers[0]>0 && (host_model_constants_integers[0] == 1 || ((bindex + 1) % host_model_constants_integers[0] == 0)) && bindex != grid.x - 1;
		int postDecoupled = host_model_constants_integers[0] > 0 && (bindex  % host_model_constants_integers[0] == 0);
		int input_sample = bindex*host_model_constants_integers[8];
		// calculates out_sample
		host_model_out_sample_index[bindex] = input_sample + std::max((1 - transient_offset - postDecoupled),0)*host_model_constants_integers[5]; // use as start output 
		// calculate end output as constant for convience
		int block_length = host_model_constants_integers[8] + (1 - preDecoupled)*host_model_constants_integers[5];
		max_block_length = max(block_length, max_block_length);
		host_model_end_sample_index[bindex] = input_sample + block_length;
		if (bindex > 0) {
			host_model_out_sample_index[bindex] = max(host_model_out_sample_index[bindex], host_model_end_sample_index[bindex-1]);
		}
		if (Show_Calculated_Power & 2) {
			std::cout << "Block[" << bindex << "] ={'start_input'=" << input_sample << ",'start_output'=" << host_model_out_sample_index[bindex] << ",'end_output'=" << host_model_end_sample_index[bindex] << "}" << std::endl;
		}
		if (Show_Calculated_Power & 4) {
			std::cout << "Block[" << bindex << "] ={'preDecoupled'=" << preDecoupled << ",'transient_offset'=" << transient_offset << ",'block_length'=" << block_length << "}" << std::endl;
		}
	}
	// long constants
	host_model_constants_longs[0] = overlapTimeMicroSec;
	// preapare symbols
	gpuAssert(cudaMemcpyToSymbol(model_constants, host_model_constants, MODEL_FLOATS_CONSTANTS_SIZE*sizeof(float)));
	gpuAssert(cudaMemcpyToSymbol(model_constants_integers, host_model_constants_integers, MODEL_INTEGERS_CONSTANTS_SIZE*sizeof(int)));
	gpuAssert(cudaMemcpy(cudaHolderGeneratedData.generated_model_out_sample_index, host_model_out_sample_index, MAX_NUMBER_OF_BLOCKS*sizeof(int),cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaHolderGeneratedData.generated_model_end_sample_index, host_model_end_sample_index, MAX_NUMBER_OF_BLOCKS*sizeof(int), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpyToSymbol(model_constants_longs, host_model_constants_longs, MODEL_LONGS_CONSTANTS_SIZE*sizeof(long)));

	// calculate convergence criteria
	if (calculate_boundary_conditions) {
		calculateBoundariesForNonGeneratedInput(
			Relative_Error_Parameters,
			max_block_length,
			Show_Calculated_Power,
			M1_SP_Fix_Factor,
			Tolerance_Fix_Factor,
			Max_M1_SP_Error_Parameter,
			Max_Tolerance_Parameter,
			host_model_constants[34],
			host_model_constants_integers[8],
			grid // tell us block partition of current run
			);
	}

	/**
	void *params[] = { cudaHolderData.cuda_input_samples,
	cudaHolderData.cuda_saved_speeds,
	cudaHolderData.cuda_Failed_Converged_Time_Node,
	cudaHolderData.cuda_Failed_Converged_Blocks,
	//cuda_saved_speeds_buffer,
	cudaHolderData.cuda_massd,
	cudaHolderData.cuda_Md,
	cudaHolderData.cuda_Ud,
	cudaHolderData.cuda_Ld,

	cudaHolderData.cuda_Rd,
	cudaHolderData.cuda_Sd,
	cudaHolderData.cuda_Qd,
	cudaHolderData.cuda_gammad,
	cudaHolderData.cuda_S_ohcd,
	cudaHolderData.cuda_S_tmd,
	cudaHolderData.cuda_R_tmd,
	cudaHolderGeneratedData.generated_model_throw_tolerance,
	cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
	cudaHolderGeneratedData.generated_model_out_sample_index,
	cudaHolderGeneratedData.generated_model_end_sample_index,
	cudaHolderData.converge_speed,
	cudaHolderData.converge_speed_blocks,
	cudaHolderData.cuda_Converged_Time_Node,
	cudaHolderData.cuda_Converged_Blocks,
	cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
	cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
	&w_ohc,
	&time_step,
	&time_step_out,
	&delta_x,
	&alpha_r,
	&enable_psi,
	&enable_OW,
	&base_time,
	&Ts,
	&_ohc_alpha_l,
	&_ohc_alpha_s,
	&model_Gme,
	&model_a0,
	&model_a1,
	&model_a2,
	&sigma_ow,
	&eta_1,
	&eta_2,
	&samplesBufferLengthP1,
	&overlap_nodes_for_block,
	&cuda_min_time_step,
	&cuda_max_time_step };
	cuLaunchKernel(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B, threads.x, threads.y, threads.z, grid.x, grid.y, grid.z, 0, NULL, params, NULL);
	*/

	// choose and execute kernel version
	printf("prefered BMOHC_Kernel_Configuration: %d\n", BMOHC_Kernel_Configuration);
	cudaEventsStartTimer(start, stop, Show_Run_Time & 1);
	if (Run_Fast_BM_Calculation == 22) {
		cudaFuncSetCacheConfig(TripleAggKernelLauncher<8>, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("TripleAggKernelLauncher<8> << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		TripleAggKernelLauncher<8> KERNEL_ARGS2(grid, threads) (
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, TripleAggKernelLauncher<8>, deviceProp);
	}
	else if (Run_Fast_BM_Calculation == 21) {
		cudaFuncSetCacheConfig(TripleAggKernelLauncher<7>, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("TripleAggKernelLauncher<7> << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		TripleAggKernelLauncher<7> KERNEL_ARGS2(grid, threads) (
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, TripleAggKernelLauncher<7>, deviceProp);
	}
	else if (Run_Fast_BM_Calculation == 20) {
		cudaFuncSetCacheConfig(TripleAggKernelLauncher<6>, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("TripleAggKernelLauncher<6> << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		TripleAggKernelLauncher<6> KERNEL_ARGS2(grid, threads) (
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, TripleAggKernelLauncher<6>, deviceProp);
	}
	else if (Run_Fast_BM_Calculation == 19) {
		cudaFuncSetCacheConfig(TripleAggKernelLauncher<5>, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("TripleAggKernelLauncher<5> << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		TripleAggKernelLauncher<5> KERNEL_ARGS2(grid, threads) (
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, TripleAggKernelLauncher<5>, deviceProp);
	} else if (Run_Fast_BM_Calculation == 18) {
		cudaFuncSetCacheConfig(TripleAggKernelLauncher<4>, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("TripleAggKernelLauncher<4> << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		TripleAggKernelLauncher<4> KERNEL_ARGS2(grid, threads) (
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
		);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, TripleAggKernelLauncher<4>, deviceProp);
	} else if (Run_Fast_BM_Calculation == 17) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations, deviceProp);
	} else if (Run_Fast_BM_Calculation == 16) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_advanced_aggregations, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_advanced_aggregations << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_advanced_aggregations KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_advanced_aggregations, deviceProp);
	} else if (Run_Fast_BM_Calculation == 15) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized, deviceProp);
	} else if (Run_Fast_BM_Calculation == 7) {
		cudaFuncSetCacheConfig(BMOHC_OLD_2017_01_13_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_OLD_2017_01_13_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_OLD_2017_01_13_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			static_cast<int>(1000000*Ts*overlap_nodes_for_block),
			1,
			cuda_max_time_step,
			cuda_min_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_OLD_2017_01_13_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 8) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_2B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_2B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_2B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_2B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 9) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_3B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_3B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_3B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_3B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 10) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 14) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_8B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_8B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_8B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_8B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 13) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_7B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_7B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_7B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_7B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 12) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_6B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_6B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}

		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_6B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_6B, deviceProp);
	} else if (Run_Fast_BM_Calculation == 11) {
		cudaFuncSetCacheConfig(BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		
		BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_Pre_fmaf_No_Constants_kernel_5B, deviceProp);
	} else  if (Run_Fast_BM_Calculation == 6) {
		cudaFuncSetCacheConfig(BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Constants_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Constants_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Constants_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			//cudaHolderData.cuda_Failed_Converged_Time_Node,
			//cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			w_ohc,
			time_step,
			time_step_out,
			delta_x,
			alpha_r,
			enable_psi,
			enable_OW,
			base_time,
			Ts,
			_ohc_alpha_l,
			_ohc_alpha_s,
			model_Gme,
			model_a0,
			model_a1,
			model_a2,
			sigma_ow,
			eta_1,
			eta_2,
			samplesBufferLengthP1,
			overlap_nodes_for_block,
			cuda_min_time_step,
			cuda_max_time_step
			//cudaHolderData.converge_speed,
			//cudaHolderData.converge_speed_blocks,
			//cudaHolderData.cuda_Converged_Time_Node,
			//cudaHolderData.cuda_Converged_Blocks,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Constants_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 5) {
		cudaFuncSetCacheConfig(BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Or_Sync_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Or_Sync_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Or_Sync_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			//cudaHolderData.cuda_Failed_Converged_Time_Node,
			//cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index
			//cudaHolderData.converge_speed,
			//cudaHolderData.converge_speed_blocks,
			//cudaHolderData.cuda_Converged_Time_Node,
			//cudaHolderData.cuda_Converged_Blocks,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_No_Or_Sync_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 4) {
		cudaFuncSetCacheConfig(BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			//cudaHolderData.cuda_Failed_Converged_Time_Node,
			//cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index
			//cudaHolderData.converge_speed,
			//cudaHolderData.converge_speed_blocks,
			//cudaHolderData.cuda_Converged_Time_Node,
			//cudaHolderData.cuda_Converged_Blocks,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_NO_SelfAnalysis_Pre_fmaf_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 3) {
		cudaFuncSetCacheConfig(BMOHC_FAST_NO_SelfAnalysis_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_NO_SelfAnalysis_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_FAST_NO_SelfAnalysis_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			//cudaHolderData.cuda_Failed_Converged_Time_Node,
			//cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index
			//cudaHolderData.converge_speed,
			//cudaHolderData.converge_speed_blocks,
			//cudaHolderData.cuda_Converged_Time_Node,
			//cudaHolderData.cuda_Converged_Blocks,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			//cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_NO_SelfAnalysis_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 2) {
		cudaFuncSetCacheConfig(BMOHC_IMPERCISE_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_IMPERCISE_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_IMPERCISE_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_IMPERCISE_kernel, deviceProp);
	} else if (Run_Fast_BM_Calculation == 1) {
		//if (deviceProp.major < 5) { // correct onlt for kepler architecture
			cudaFuncSetCacheConfig(BMOHC_FAST_kernel, static_cast<cudaFuncCache>(BMOHC_Kernel_Configuration));
		//}
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_FAST_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_FAST_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			//cuda_saved_speeds_buffer,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
			);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_FAST_kernel, deviceProp);
	} else {
		if (Show_Calculated_Power & 16) {
			printf("BMOHC_NEW_kernel << <%s,%s>>>(...);\n", showDIM3(grid).c_str(), showDIM3(threads).c_str());
		}
		BMOHC_NEW_kernel KERNEL_ARGS2(grid, threads)(
			cudaHolderData.cuda_input_samples,
			cudaHolderData.cuda_saved_speeds,
			cudaHolderData.cuda_Failed_Converged_Time_Node,
			cudaHolderData.cuda_Failed_Converged_Blocks,
			cudaHolderData.cuda_massd,
			cudaHolderData.cuda_Md,
			cudaHolderData.cuda_Ud,
			cudaHolderData.cuda_Ld,

			cudaHolderData.cuda_Rd,
			cudaHolderData.cuda_Sd,
			cudaHolderData.cuda_Qd,
			cudaHolderData.cuda_gammad,
			cudaHolderData.cuda_S_ohcd,
			cudaHolderData.cuda_S_tmd,
			cudaHolderData.cuda_R_tmd,
			cudaHolderGeneratedData.generated_model_throw_tolerance,
			cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance,
			cudaHolderGeneratedData.generated_model_out_sample_index,
			cudaHolderGeneratedData.generated_model_end_sample_index,
			cudaHolderData.converge_speed,
			cudaHolderData.converge_speed_blocks,
			cudaHolderData.cuda_Converged_Time_Node,
			cudaHolderData.cuda_Converged_Blocks,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration,
			cudaHolderData.cuda_convergence_jacoby_loops_per_iteration_blocks
		);
		if (Show_Device_Data & 32) cudaOccupancyIndicator(threads.x, BMOHC_NEW_kernel, deviceProp);
	}
    
	std::ostringstream oss("");
	oss << "BM (" << run_mode_name << ") run time";
	
	outer_log.timeAtFlag(33, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 1, oss.str()), Show_Run_Time & 1);
	gpuAssert(cudaDeviceSynchronize());
    // copy results to host
	//std::cout << "passed kernel...\n";
    cutilCheckMsg("OHCBM_kernel<<<>>> execution failed\n");
}
extern "C" void cudaMallocatingByMode(void **ptr,size_t bytes_num,bool disable_advanced_mode) noexcept(false) {
	if (disable_advanced_mode) { gpuAssert(cudaMalloc(ptr, bytes_num)); }
	else {
		gpuAssert(cudaMallocManaged(ptr, bytes_num));
	}
} 


/* IHC combined variables */
size_t input_max_size;
size_t input_max_size_in_bytes;
size_t double_input_max_size_in_bytes;
size_t lambda_float_input_max_size_in_bytes;
size_t lambda_max_size;
size_t lambda_max_size_in_bytes;
size_t lambda_double_max_size_in_bytes;
size_t lambda_forced_double_max_size_in_bytes;
size_t backup_speeds_size_in_bytes;
size_t matrixSizeOfBackupTime; // ensure position of taking the backup array
size_t matrixSizeOfBackupTimeInBytes; // ensure position of taking the backup array
size_t matrixSizeOfCalcTime; // ensure position of taking the backup array
size_t matrixSizeOfCalcTimeInBytes; // ensure position of taking the backup array
size_t matrixSizeOfWriteTime;
size_t matrixSizeOfWriteTimeInBytes;
bool localloadedFromHD;
bool local_first_time_for_param_set;
bool local_CalculateJNDOnGPU;
float *local_backup_speeds;
size_t local_backup_speeds_length;
/**
* ac filter device includes its size in first place
*/
__constant__ double device_time_filter[DEVICE_MAX_FILTER_ORDER];

/**
* ihc damage vector (nerves converted to 0 to 10^8)
*/
__constant__ double CUDA_IHC_DAMAGE[SECTIONS];


/**
* nerves cluster parmeters, A and W and spont  (nerves converted to 0 to 10^8)
*/
__constant__ float CUDA_Nerves_Clusters[3*LAMBDA_COUNT];

/**
* ac filters  includes its size in first place
*/
double host_filter[DEVICE_MAX_FILTER_ORDER];

/*end of ihc combined variables */
#define ORDER_OF_DC_FILTER_SIZE_IN_PARAMS 4
#define ORDER_OF_AC_FILTER_SIZE_IN_PARAMS 5

extern "C" void InputProfilesArrayInitialization(
	int maxJNDIntervals,
	int wn_length,
	int signal_length,
	int signal_mode,
	int Show_Generated_Input
	) {
	cudaJNDHolder.allocateIntervals(maxJNDIntervals);
	if ( wn_length > 0 ) {
		cudaSignalHolder.allocateSignalNodes(wn_length);
		if (Show_Generated_Input & 8) std::cout << "WN array allocated length: " << wn_length << std::endl;
		
	}
}

// release input array
extern "C" void InputProfilesArrayTermination() {
	cudaJNDHolder.releaseIntervals();
	cudaSignalHolder.releaseSignalNodes();
}
// calculates IHC see description in cochlea_common.h
extern "C" void IHCNewKernel(
	double *IHC_Damage_Factor,
	float Nerves_Clusters[3 * LAMBDA_COUNT],
	double *dc_filter,
	int order_of_dc_filter,
	double *ac_b_filter,
	double *ac_a_filter,
	bool is_ac_iir_filter,
	int order_of_ac_filter,
	int cochlea_sections,
	int time_blocks,
	double SPLRefVal,
	float *backup_speeds,
	int backup_speeds_length,
	int calcTime,
	int writeTime,
	int allocateTime,
	int intervalTimeNodes, // single time block time nodes
	int max_backup_nodes_len,
	int lambda_offset, // offset of time nodes in order to compensate for larger lambda than necessary
	float Lambda_SAT,
	float eta_AC, // IHC AC coupling [V/s/cm]
	float eta_DC, // IHC DC coupling [V/s/cm]
	bool first_time,
	bool first_time_for_param_set,
	bool loadedFromHD,
	bool disable_advanced_memory_handling,
	bool review_memory_handling,
	bool asMemoryHandlingOnly,
	float scaleBMVelocityForLambdaCalculation, // params[params_set_counter].scaleBMVelocityForLambdaCalculation
	bool CalculateJNDOnGPU,
	int maxJNDIntervals,
	int overlapNodes,
	int Decouple_Filter,	 // filter is decoupled if this parameter largeer than 0,if filter decoupled than output blocks will not use input with time start before output block start		
	int Show_Run_Time,
	Log &outer_log
	) noexcept(false) {
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 4);
	local_CalculateJNDOnGPU = CalculateJNDOnGPU;
	// calculate sizes of Input and output arrays
	input_max_size = allocateTime*cochlea_sections + max_backup_nodes_len;
	int mean_size = maxJNDIntervals*LAMBDA_COUNT*cochlea_sections;
	int fisher_size = maxJNDIntervals*LAMBDA_COUNT;
	int dA_size = maxJNDIntervals;
		input_max_size_in_bytes = input_max_size*sizeof(float);
		double_input_max_size_in_bytes = input_max_size*sizeof(double);
		lambda_float_input_max_size_in_bytes = input_max_size*sizeof(lambdaFloat);
		lambda_max_size = LAMBDA_COUNT*input_max_size;
		//std::cout << "allocated " << lambda_max_size << " nodes for calculation\n";
		lambda_max_size_in_bytes = lambda_max_size*sizeof(float);
		lambda_double_max_size_in_bytes = lambda_max_size*sizeof(lambdaFloat);
		lambda_forced_double_max_size_in_bytes = lambda_max_size*sizeof(JNDFloat);
		matrixSizeOfBackupTime = calcTime*cochlea_sections - backup_speeds_length;
		matrixSizeOfBackupTimeInBytes = matrixSizeOfBackupTime*sizeof(float);
		matrixSizeOfCalcTime = calcTime*cochlea_sections;
		matrixSizeOfCalcTimeInBytes = matrixSizeOfBackupTime*sizeof(float);
		backup_speeds_size_in_bytes = max_backup_nodes_len*sizeof(float);
		matrixSizeOfWriteTime = writeTime*cochlea_sections;
		matrixSizeOfWriteTimeInBytes = matrixSizeOfWriteTime*sizeof(float);
		local_backup_speeds = backup_speeds;
		local_first_time_for_param_set = first_time_for_param_set;
		localloadedFromHD = loadedFromHD;
		local_backup_speeds_length = backup_speeds_length;
		int size_of_device_params = sizeof(device_params);
		int size_of_vectors_sum_linear_coefficents = sizeof(vectors_sum_linear_coefficents);
		vectors_sum_linear_coefficents host_vectors_sum_linear_coefficents[2];
		
		for ( int i=0;i<DEVICE_MAX_FILTER_ORDER;i++){
			host_filter[i] = 0;
		}
		// allocate AC/DC filters on GPU and copy them
		host_filter[0] = (double)order_of_dc_filter;
		int ac_filter_b_index = (int)host_filter[0]+1;
		int ac_filter_a_index = -1;
		host_filter[ac_filter_b_index] = (double)order_of_ac_filter;
		memcpy_s(&host_filter[1],sizeof(double)*(DEVICE_MAX_FILTER_ORDER-1),dc_filter,order_of_dc_filter*sizeof(double));
		memcpy_s(&host_filter[ac_filter_b_index + 1], sizeof(double)*(DEVICE_MAX_FILTER_ORDER - order_of_dc_filter - 2), ac_b_filter, order_of_ac_filter*sizeof(double));
		std::cout.precision(5);
		// update ac filter for auto scale of 	scaleBMVelocityForLambdaCalculation
		for (int ix = 0; ix < order_of_ac_filter; ix++) {
			host_filter[ac_filter_b_index + 1 + ix] = scaleBMVelocityForLambdaCalculation*host_filter[ac_filter_b_index + 1 + ix];
			//std::cout << "host_filter[" << (ac_filter_b_index + ix + 1) << "] = " << host_filter[ac_filter_b_index + 1 + ix] << "\n";
		}
		if ( is_ac_iir_filter) {
			ac_filter_a_index = ac_filter_b_index + order_of_ac_filter + 1;
			host_filter[ac_filter_a_index] = (double)order_of_ac_filter;
			memcpy_s(&host_filter[ac_filter_a_index + 1], sizeof(double)*(DEVICE_MAX_FILTER_ORDER - order_of_dc_filter - order_of_ac_filter - 3), ac_a_filter, order_of_ac_filter*sizeof(double));
			/*for (int ix = 0; ix < order_of_ac_filter; ix++) {
				std::cout << "host_filter[" << (ac_filter_a_index + ix + 1) << "] = " << host_filter[ac_filter_a_index + 1 + ix] << "\n";
			}*/
		}
		if (review_memory_handling) {
			printf("Lambda Memory Size %lu KB\n", (lambda_max_size_in_bytes / 1024));
			printf("calcTime %ld Nodes\n", calcTime);
			printf("Input (Results BM Speeds) Allocated Memory Size %lu KB\n", (input_max_size_in_bytes / 1024));
			printf("Input (Results BM Speeds) Array Allocated length %lu \n ", input_max_size);
			printf("Allocate time %ld \n", allocateTime);
			printf("Backup speeds size %lu KB\n ",(backup_speeds_size_in_bytes / 1024));
		}
		// allocate lambda memory
		cudaLambdaHolderData.allocateLambdaMemory(input_max_size);
		cudaJNDHolder.allocateDeviceStructs();
		cudaEventsStartTimer(start, stop, Show_Run_Time & 4);
			
		// allocate JND intermidiate arrays (Eq. 17-24 in Cochlear Model for Hearing Loss)	
		if (CalculateJNDOnGPU) {
			cudaJNDHolder.allocateMeanNodes(mean_size);


			cudaLambdaHolderData.allocateIHCData();
			cudaJNDHolder.allocateIntervals(dA_size);
			cudaJNDHolder.allocateFisherNodes(fisher_size);
				
		}
		
		outer_log.timeAtFlag(38, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 4, "Initialize and allocate Memory for Lambda Calculation run"), Show_Run_Time & 4);
		if (!asMemoryHandlingOnly)	   {
			// copy all IHC parmeters necessary due to changed profile to GPU
			cudaEventsStartTimer(start, stop, Show_Run_Time & 4);
			if (first_time_for_param_set) {
				/**
				* define the coefficents for both vectors sums
				* at node 0 A_coefficent=1,B_coeffient=-1 for SHigh = BM_input - AC
				* at node 1 A_coefficent=eta_AC,B_coeffient=eta_DC for IHC=eta_AC*AC+eta_DC*DC
				*/
				host_vectors_sum_linear_coefficents[0].A_coefficent = scaleBMVelocityForLambdaCalculation;
				host_vectors_sum_linear_coefficents[0].B_coefficent = -1;
				host_vectors_sum_linear_coefficents[0].reverseCoefficents = 0;
				host_vectors_sum_linear_coefficents[1].A_coefficent = eta_AC;// (eta_AC*scaleBMVelocityForLambdaCalculation);
				host_vectors_sum_linear_coefficents[1].B_coefficent = eta_DC;// (eta_DC*scaleBMVelocityForLambdaCalculation*scaleBMVelocityForLambdaCalculation);
				host_vectors_sum_linear_coefficents[1].reverseCoefficents = 0;
				if (host_vectors_sum_linear_coefficents[1].B_coefficent < 1 && sizeof(host_vectors_sum_linear_coefficents[1].B_coefficent) == sizeof(float)) {
					host_vectors_sum_linear_coefficents[1].reverseCoefficents = 1;
					host_vectors_sum_linear_coefficents[1].A_coefficent = 1.0 / host_vectors_sum_linear_coefficents[1].A_coefficent;
					host_vectors_sum_linear_coefficents[1].B_coefficent = 1.0 / host_vectors_sum_linear_coefficents[1].B_coefficent;
				}
	

				gpuAssert(cudaMemcpy(cudaJNDHolder.vectors_sums_coefficents, host_vectors_sum_linear_coefficents, 2 * size_of_vectors_sum_linear_coefficents, cudaMemcpyHostToDevice));
				// copy ihc damage factor
				gpuAssert(cudaMemcpyToSymbol(CUDA_IHC_DAMAGE, IHC_Damage_Factor, SECTIONS*sizeof(double), 0, cudaMemcpyHostToDevice));
				// copy nerves parameters
				gpuAssert(cudaMemcpyToSymbol(CUDA_Nerves_Clusters, Nerves_Clusters, 3 * LAMBDA_COUNT*sizeof(float), 0, cudaMemcpyHostToDevice));
				gpuAssert(cudaMemcpyToSymbol(device_time_filter, host_filter, DEVICE_MAX_FILTER_ORDER*sizeof(double), 0, cudaMemcpyHostToDevice));

				// here I prime and load to the device both the filters and and the genral params

			}
			cudaJNDHolder.host_local_param[0].Decouple_Filter = Decouple_Filter;
			cudaJNDHolder.host_local_param[0].cochlea_sections = cochlea_sections;
			cudaJNDHolder.host_local_param[0].intervalTimeNodes = intervalTimeNodes;
			cudaJNDHolder.host_local_param[0].time_blocks = time_blocks;
			cudaJNDHolder.host_local_param[0].ovelapNodes = overlapNodes;
			cudaJNDHolder.host_local_param[0].lambda_offset = lambda_offset;
			cudaJNDHolder.host_local_param[0].order_of_dc_filter = order_of_dc_filter;
			cudaJNDHolder.host_local_param[0].order_of_ac_filter = order_of_ac_filter;
			cudaJNDHolder.host_local_param[0].lambda_count = LAMBDA_COUNT;
			cudaJNDHolder.host_local_param[0].time_block = calcTime / time_blocks;
			cudaJNDHolder.host_local_param[0].FilterDecoupledMode = Decouple_Filter>0;
			cudaJNDHolder.host_local_param[0].reverseSQRTScaleBMVelocityForLambdaCalculation = 1 / sqrtf(scaleBMVelocityForLambdaCalculation); // necessary adjustment for dA fix
			//std::cout << "reverseSQRTScaleBMVelocityForLambdaCalculation = " << cudaJNDHolder.host_local_param[0].reverseSQRTScaleBMVelocityForLambdaCalculation << "\n";
			//std::cout << "prepares host local setups" << std::endl;
			cudaJNDHolder.host_local_param[0].Lambda_SAT = Lambda_SAT;
			cudaJNDHolder.host_local_param[0].calcTime = calcTime;
			//std::cout << "cudaJNDHolder.host_local_param[0].intervalTimeNodes*host_local_params[0].time_blocks= " << (cudaJNDHolder.host_local_param[0].intervalTimeNodes*host_local_params[0].time_blocks) << "\n";
			cudaJNDHolder.host_local_param[0].writeTime = writeTime;
			cudaJNDHolder.host_local_param[0].filter_b_start_index = ac_filter_b_index; // first filter is ac 
			cudaJNDHolder.host_local_param[0].filter_a_start_index = ac_filter_a_index; // first filter is ac 		  
			/*std::cout << "cudaJNDHolder.host_local_param[0].filter_b_start_index: " << cudaJNDHolder.host_local_param[0].filter_b_start_index << "\n";
			std::cout << "cudaJNDHolder.host_local_param[0].filter_a_start_index: " << cudaJNDHolder.host_local_param[0].filter_a_start_index << "\n";*/
			//std::cout << "prepares host local copy" << std::endl;
			memcpy_s(&cudaJNDHolder.host_local_param[1], size_of_device_params, &cudaJNDHolder.host_local_param[0], size_of_device_params);
			cudaJNDHolder.host_local_param[1].filter_b_start_index = 0; // second filter is dc	
			cudaJNDHolder.host_local_param[1].filter_a_start_index = -1; // second filter is dc
			//std::cout << "prepares gpu copy" << std::endl;
			gpuAssert(cudaMemcpy(cudaJNDHolder.global_device_params, cudaJNDHolder.host_local_param, 2 * size_of_device_params, cudaMemcpyHostToDevice));
			//std::cout << "passed allocations sequence..." << std::endl;		 
			
			outer_log.timeAtFlag(39, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 4, "Load Parmeters for Lambda Calculation run"), Show_Run_Time & 4);
		}
}
// allocate GPU buffer
extern "C" void allocateBuffer(const int size_in_nodes, int Show_Run_Time, cudaEvent_t& start,
	cudaEvent_t& stop,
	cudaDeviceProp deviceProp)  noexcept(false) {
	cudaEventsCreate(start, stop, Show_Run_Time & 32);
	std::ostringstream oss("");
	oss << "Allocate Debug Buffer (" << size_in_nodes << " Nodes)";
	std::string rec = oss.str();
	cudaEventsStartTimer(start, stop, Show_Run_Time & 32);
	cudaLambdaHolderData.allocateBufferMemory(size_in_nodes);
	
	cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 32, rec);
}
// release GPU buffer
extern "C" void releaseBuffer( int Show_Run_Time, cudaEvent_t& start,
	cudaEvent_t& stop,
	cudaDeviceProp deviceProp)  noexcept(false) {
	
		cudaEventsCreate(start, stop, Show_Run_Time & 32);
		cudaEventsStartTimer(start, stop, Show_Run_Time & 32);
		cudaLambdaHolderData.releaseBufferMemory();
		cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 32, "release Debug Buffer");
}
extern "C" void IHCKernel_Free()  noexcept(false) {
	cudaLambdaHolderData.releaseLambdaMemory();
	cudaJNDHolder.releaseIntervals(); 
	cudaJNDHolder.ReleaseDeviceStructs();
	cudaJNDHolder.releaseFisherNodes();
	cudaJNDHolder.releaseMeanNodes();
}
/**
* calculate correct section for the thread to handle out of cuda parameters of calling function
* number of sections per block mult by block number + thread id per section in the block
* blockCoordinates=blockIdx
* blocksDimensions=blockDim
* threadCoordinates=threadIdx
*/
#define calcCochleaSection(blockCoordinates,blocksDimensions,threadCoordinates) (blocksDimensions.x*blockCoordinates.x + threadCoordinates.x)

/**
* calculate number of time nodes each thread will jump every consecutive calculated point
* block dimension y represents number of threads working on the same section
* blockCoordinates=blockIdx
* blocksDimensions=blockDim
* threadCoordinates=threadIdx
*/
#define calcTimeNodesJump(blockCoordinates,blocksDimensions,threadCoordinates) (blocksDimensions.y)


/**
* calculate the lambda cluster for the thread by blockIdx.y==blockCoordinates.y
* blockCoordinates=blockIdx
* blocksDimensions=blockDim
* threadCoordinates=threadIdx
*/
#define calcLambdaBlock(blockCoordinates,blocksDimensions,threadCoordinates) (blockCoordinates.y)


/**
* calculate number of time nodes offset from the begginning for the thread to work on
* thread dimension y represent thread id of working in the same section
* blockCoordinates=blockIdx
* blocksDimensions=blockDim
* threadCoordinates=threadIdx
*/
#define calcTimeNodesOffset(blockCoordinates,blocksDimensions,threadCoordinates) (threadCoordinates.y)



/**
* calculate start offset per thread on unified calculations of indexes
* blockCoordinates=blockIdx
* blocksDimensions=blockDim
* threadCoordinates=threadIdx
*/
#define calcStartMainOffset(blockCoordinates,blocksDimensions,threadCoordinates,lambda_offset,cochlea_sections) (calcCochleaSection(blockCoordinates,blocksDimensions,threadCoordinates)+ cochlea_sections*lambda_offset)


/**
* decoupled block id
*/
#define intervalDecoupledBlockId(blockCoordinates,blocksDimensions) (blockCoordinates.z+blocksDimensions.z*blockCoordinates.y)

/**
* number of decoupled blocks per interval for unified IHC/ Lambda calculation
*
*/
#define intervalDecoupledBlocks(blocksDimensions) (blocksDimensions.z*blocksDimensions.y)

#define totalDecoupledBlocks(blocksDimensions) (blocksDimensions.x*intervalDecoupledBlocks(blocksDimensions))

/**
* decoupled block id
*/
#define decoupledBlockId(blockCoordinates,blocksDimensions) (intervalDecoupledBlockId(blockCoordinates,blocksDimensions)+blockCoordinates.x*intervalDecoupledBlocks(blocksDimensions))


/**
* device run for single time block of single index
* note offset here is the offset for start of time node name
* \p X is global input array
* \p Y is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p filter_size is order of FIR filter
* \p filter_index is position on unfied device filter array to start the filter from
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
* \p singleBlockLength in case of decoupled mode will be able to ignore "tails" on necessary positions,  time length analysis for not decoupled mode
*
*/

template<typename T1, typename T2> __device__ void DeviceCudaFIRFilter(T1 *X, T2 *Y, int offset, int time_length_analysis, int cochlea_sections, int filter_size, int filter_index, int time_node_offset, int time_node_jumps_in_cluster, int singleBlockLength, int final_regular_division_position) {
	int k = 0;
	int i = 0;
	int current_offset; // progress on the input since we jump in sections every time
	int offset_boundary;
	double Ycurrent;
	int sgny = 0;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		sgny = (k / (final_regular_division_position - 1));
		sgny = sgny*time_length_analysis + (1 - sgny)*singleBlockLength;
		offset_boundary = min((k + 1) % sgny, filter_size); // note if  singleBlockLength == 	time_length_analysis
		Ycurrent = 0.0f;
		for (i = 0; i<offset_boundary; i++) {
			Ycurrent = Ycurrent + device_time_filter[i + filter_index] * X[current_offset - i*cochlea_sections]; // untransposed jumping by sections each time
		}
		Y[current_offset] = T2(Ycurrent);
	}
}

/**
* device run for single sesction
* \p X is global input array
* \p Y is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p filter_size is order of FIR filter
* \p filter_a_index is position on unfied device filter array to start the a coefficents of filter
* \p filter_b_index is position on unfied device filter array to start the b coefficents of filter
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<class T1, class T2> __device__ void DeviceCudaIIRFilter(T1 *X,T2 *Y, int offset, int time_length_analysis, int IntervalLength, int final_regular_division_position, int cochlea_sections, int filter_size, int filter_b_index, int filter_a_index)
{
	int k = 0;
	int i = 0;
	int j;
	int offset_boundarya;
	int current_offset; // progress on the input since we jump in sections every time
	int offset_boundary;
	T2 Ycurrent = T2(0.0);
	int sgny = 0;
	for (k = 0; k<time_length_analysis; k++) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		sgny = (k / (final_regular_division_position - 1));
		sgny = sgny*time_length_analysis + (1 - sgny)*IntervalLength;
		//offset_boundary = min((k + 1) % ((1-sgny)*IntervalLength + 2*sgny*time_length_analysis), filter_size);
		offset_boundary = min((k + 1) % sgny, filter_size);
		offset_boundarya = offset_boundary - 1;
		Ycurrent = T2(0.0);
		for (i = 0; i<offset_boundary; i++) {
			//Ycurrent = fmaf( device_time_filter[i + 1 + filter_b_index], X[current_offset - i*cochlea_sections], Ycurrent);
			Ycurrent = Ycurrent + device_time_filter[i + filter_b_index] * X[current_offset - i*cochlea_sections]; // untransposed jumping by sections each time
		}
		for (i = 0; i<offset_boundarya; i++) {
			j = i + 1;
			//Ycurrent = fmaf(-1 * device_time_filter[j + 1 + filter_a_index], Y[current_offset - j*cochlea_sections], Ycurrent);
			Ycurrent = Ycurrent - device_time_filter[j + filter_a_index] * Y[current_offset - j*cochlea_sections]; // untransposed jumping by sections each time
		}
		Y[current_offset] = Ycurrent;
	}
}



/**
* device run for single sesction
* \p X is global input array
* \p Y is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p filter_size is order of FIR filter
* \p filter_a_index is position on unfied device filter array to start the a coefficents of filter
* \p filter_b_index is position on unfied device filter array to start the b coefficents of filter
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<class T1,class T2> __host__ void DeviceCudaIIRFilterHost(T1 *X, T2 *Y, int offset, int time_length_analysis, int cochlea_sections, int filter_size, int filter_b_index, int filter_a_index) {
	int k = 0;
	int i = 0;
	int j;
	int offset_boundarya;
	int current_offset; // progress on the input since we jump in sections every time
	int offset_boundary;
	double Ycurrent = 0.0;
	for (k = 0; k<time_length_analysis; k++) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		offset_boundary = __tmin(k + 1, filter_size);
		offset_boundarya = offset_boundary - 1;
		Ycurrent = 0.0f;
		for (i = 0; i<offset_boundary; i++) {
			//Ycurrent = fmaf( device_time_filter[i + 1 + filter_b_index], X[current_offset - i*cochlea_sections], Ycurrent);
			Ycurrent = Ycurrent + host_filter[i + filter_b_index] * X[current_offset - i*cochlea_sections]; // untransposed jumping by sections each time
		}
		for (i = 0; i<offset_boundarya; i++) {
			j = i + 1;
			//Ycurrent = fmaf(-1 * device_time_filter[j + 1 + filter_a_index], Y[current_offset - j*cochlea_sections], Ycurrent);
			Ycurrent = Ycurrent - host_filter[j + filter_a_index] * Y[current_offset - j*cochlea_sections]; // untransposed jumping by sections each time
		}
		Y[current_offset] = T2(Ycurrent);
	}
}

// possible types for iir on host
template __host__ void DeviceCudaIIRFilterHost<float, double>(float *X, double *Y, int offset, int time_length_analysis, int cochlea_sections, int filter_size, int filter_b_index, int filter_a_index);
template __host__ void DeviceCudaIIRFilterHost<double, double>(double *X, double *Y, int offset, int time_length_analysis, int cochlea_sections, int filter_size, int filter_b_index, int filter_a_index);
template __host__ void DeviceCudaIIRFilterHost<float, float>(float *X, float *Y, int offset, int time_length_analysis, int cochlea_sections, int filter_size, int filter_b_index, int filter_a_index);

// this version runs the entire time line on relevant sections
template<typename T1,typename T2> __global__ void CudaFIRFilter(T1 *X, T2 *Y, device_params *filter_params)
{
	int filter_index = filter_params->filter_b_start_index+1;
	int filter_size = (int)(device_time_filter[filter_index-1]+0.1f); // very important filter data start from index 1 0 index is size....
	//if (threadIdx.x == 0) printf("filter_index=%d,filter_size=%d\n", filter_index, filter_size);
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int total_time_nodes = (filter_params->calcTime - lambda_offset);
	int time_length_analysis = filter_params->intervalTimeNodes/gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int grid_block_id = (decoupledBlockId(blockIdx, gridDim) - intervalDecoupledBlockId(blockIdx, gridDim));																																  // each thread start from its own adjusted offset in the time block offset
	int offset = cochlea_offset_section + cochlea_sections*(grid_block_id*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
	int time_node_offset = intervalDecoupledBlockId(blockIdx,gridDim)*time_length_analysis;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	time_length_analysis = time_node_offset + time_length_analysis;
	// on stage one device filter run from global
	int calculatedIntervalTimeNodes = filter_params->FilterDecoupledMode ? filter_params->intervalTimeNodes*filter_params->Decouple_Filter : total_time_nodes;
	int final_regular_division_position = filter_params->intervalTimeNodes *gridDim.y;
	
	//if (threadIdx.x==0) printf("block[%d].time_nodes=[%d,%d],interval_offset=%d,grid_block_id=%d,final_regular_division_position=%d\n", decoupledBlockId(blockIdx, gridDim), time_node_offset, time_length_analysis,offset, grid_block_id, final_regular_division_position);
	DeviceCudaFIRFilter<T1, T2>(X, Y, offset, time_length_analysis, cochlea_sections, filter_size, filter_index, time_node_offset, time_node_jumps_in_cluster, calculatedIntervalTimeNodes, final_regular_division_position);
	__syncthreads();
}


// cuda fir filter
template __global__ void CudaFIRFilter<float, double>(float *X, double *Y, device_params *filter_params);
template __global__ void CudaFIRFilter<double, double>(double *X, double *Y, device_params *filter_params);

// this version runs the entire time can be aprralleized by sections
template<class T1,class T2> __global__ void CudaIIRFilter(T1 *X, T2 *Y, device_params *filter_params) {
	int filter_b_index = 1 + filter_params->filter_b_start_index;
	int filter_a_index = 1 + filter_params->filter_a_start_index;
	int filter_size = filter_params->order_of_ac_filter;//(int)(device_time_filter[filter_b_index-1]+0.1); // very important filter data start from index 1 0 index is size....
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->calcTime - lambda_offset; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	/*
	*  start offset represents calculated section + lambda offset_in_time*sections +
	*  time offset per thread per section since each thread handles 1/blockDim.y
	*/
	// on stage one device filter run from global
	int intervalLength = filter_params->FilterDecoupledMode == true ? filter_params->intervalTimeNodes*filter_params->Decouple_Filter : time_length_analysis;
	int final_regular_division_position = filter_params->intervalTimeNodes*filter_params->time_blocks;
	int startOffset = threadIdx.x;
	DeviceCudaIIRFilter<T1, T2>(X, Y, startOffset, time_length_analysis, intervalLength, final_regular_division_position, cochlea_sections, filter_size, filter_b_index, filter_a_index);
	
	__syncthreads();
}

// cuda iir filter
template __global__ void CudaIIRFilter<float, double>(float *X, double *Y, device_params *filter_params);
template __global__ void CudaIIRFilter<float, float>(float *X, float *Y, device_params *filter_params);
template __global__ void CudaIIRFilter<double, double>(double *X, double *Y, device_params *filter_params);

/**
* this version runs multiple decoupled IIR filters for small inputs
*/
template<class T1, class T2> __global__ void CudaIIRFilterDecoupled(T1 *X, T2 *Y, device_params *filter_params) {
	int filter_b_index = 1 + filter_params->filter_b_start_index;
	int filter_a_index = 1 + filter_params->filter_a_start_index;
	int filter_size = filter_params->order_of_ac_filter;//(int)(device_time_filter[filter_b_index-1]+0.1); // very important filter data start from index 1 0 index is size....
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->calcTime - lambda_offset; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	/*
	*  start offset represents calculated section + lambda offset_in_time*sections +
	*  time offset per thread per section since each thread handles 1/blockDim.y
	*/
															
	// on stage one device filter run from global
	int intervalLength = filter_params->intervalTimeNodes*filter_params->Decouple_Filter;
	int final_regular_division_position = filter_params->intervalTimeNodes*filter_params->time_blocks;
	int startOffset = threadIdx.x + blockIdx.x*intervalLength*cochlea_sections;
	//printf("startOffset: %d,threadIdx.x=%d,blockIdx.x=%d,intervalLength=%d\n", startOffset, threadIdx.x, blockIdx.x, intervalLength);
	if (blockIdx.x+1==gridDim.x) {
		intervalLength += (time_length_analysis - gridDim.x*intervalLength);
	}
	time_length_analysis = intervalLength;
	DeviceCudaIIRFilter<T1, T2>(X, Y, startOffset, time_length_analysis, intervalLength, final_regular_division_position, cochlea_sections, filter_size, filter_b_index, filter_a_index);
	__syncthreads();
}

// cuda iir filter
template __global__ void CudaIIRFilterDecoupled<float, double>(float *X, double *Y, device_params *filter_params);
template __global__ void CudaIIRFilterDecoupled<float, float>(float *X, float *Y, device_params *filter_params);
template __global__ void CudaIIRFilterDecoupled<double, double>(double *X, double *Y, device_params *filter_params);

// this version runs the entire time can be aprralleized by sections
template<class T1,class T2> __host__ void CudaIIRFilterHost(T1 *X, T2 *Y, device_params *filter_params) {
	int filter_b_index = 1 + filter_params->filter_b_start_index;
	int filter_a_index = 1 + filter_params->filter_a_start_index;
	int filter_size = filter_params->order_of_ac_filter;//(int)(device_time_filter[filter_b_index-1]+0.1); // very important filter data start from index 1 0 index is size....
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->calcTime - lambda_offset; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	/*
	*  start offset represents calculated section + lambda offset_in_time*sections +
	*  time offset per thread per section since each thread handles 1/blockDim.y
	*/
	for (int tidx = 0; tidx < cochlea_sections; tidx++) {
		DeviceCudaIIRFilterHost<T1,T2>(X, Y, tidx, time_length_analysis, cochlea_sections, filter_size, filter_b_index, filter_a_index);
	}
}



template __host__ void CudaIIRFilterHost<float, float>(float *X, float *Y, device_params *filter_params);
template __host__ void CudaIIRFilterHost<float, double>(float *X, double *Y, device_params *filter_params);
template __host__ void CudaIIRFilterHost<double, double>(double *X, double *Y, device_params *filter_params);

/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p A is first global input array
* \p B is second global input array
* \p C is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p coefficentA is coefficent to multiply vector A
* \p coefficentB is coefficent to multiply vector B
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T1, typename T2> __device__ void DeviceCudaVectorSum(T1 *A, T2 *B, T2 *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster) {
	int k;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		C[current_offset] = (A[current_offset] * coefficentA) + (B[current_offset] * coefficentB);
	}
}

// unified IHC divide of types
template __device__ void DeviceCudaVectorSum<float, double>(float *A, double *B, double *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);
template __device__ void DeviceCudaVectorSum<float, float>(float *A, float *B, float *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);


/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p A is first global input array
* \p B is second global input array
* \p C is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p coefficentA is coefficent to multiply vector A
* \p coefficentB is coefficent to multiply vector B
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T1, typename T2> __device__ void DeviceCudaVectorSumNSquare(T1 *A, T2 *B, T2 *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster) {
	int k;
	T2 midSum;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		midSum = fmaf(A[current_offset], coefficentA, B[current_offset] * coefficentB);
		C[current_offset] = midSum*midSum;
	}
}


// unified IHC divide of types
template __device__ void DeviceCudaVectorSumNSquare<float, double>(float *A, double *B, double *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);
template __device__ void DeviceCudaVectorSumNSquare<float, float>(float *A, float *B, float *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);

/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p A is first global input array
* \p B is second global input array
* \p C is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p coefficentA is coefficent to multiply vector A
* \p coefficentB is coefficent to multiply vector B
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T1,typename T2> __device__ void DeviceCudaVectorSumDivide(T1 *A, T2 *B, T2 *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster) {
	int k;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		C[current_offset] = (A[current_offset] / coefficentA) + (B[current_offset] / coefficentB);
	}
}



// same for divide templae
template __device__ void DeviceCudaVectorSumDivide<float, double>(float *A, double *B, double *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);
template __device__ void DeviceCudaVectorSumDivide<float, float>(float *A, float *B, float *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);

/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p A is first global input array
* \p B is second global input array
* \p C is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p coefficentA is coefficent to multiply vector A
* \p coefficentB is coefficent to multiply vector B
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T1, typename T2> __device__ void DeviceCudaVectorSumDivideNSquare(T1 *A, T2 *B, T2 *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster) {
	int k;
	int current_offset;
	T2 midSum;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		midSum = (A[current_offset] / coefficentA) + (B[current_offset] / coefficentB);
		C[current_offset] = midSum*midSum;
	}
}



// same for divide templae
template __device__ void DeviceCudaVectorSumDivideNSquare<float, double>(float *A, double *B, double *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);
template __device__ void DeviceCudaVectorSumDivideNSquare<float, float>(float *A, float *B, float *C, int offset, int time_length_analysis, int cochlea_sections, double coefficentA, double coefficentB, int time_node_offset, int time_node_jumps_in_cluster);



/**
* calcs A*coefficents_set->A_coefficent+B*coefficents_set->B_coefficent => C for calculating shigh and ac+dc summary created with consistent indexes calculations
*
*/
template<typename T1, typename T2> __global__ void CudaVectorsSum(T1 *A, T2 *B, T2 *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set)
{
	lambdaFloat coefficentA = lambdaFloat(coefficents_set->A_coefficent);
	lambdaFloat coefficentB = lambdaFloat(coefficents_set->B_coefficent);
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	if (coefficents_set->reverseCoefficents) {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSumDivide(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaVectorSumDivide(A, B, C, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}
	else {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSum(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaVectorSum(A, B, C, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}

	__syncthreads();
}



// cuda vector sum
template __global__ void CudaVectorsSum<float, double>(float *A, double *B, double *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);
template __global__ void CudaVectorsSum<float, float>(float *A, float *B, float *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);
template __global__ void CudaVectorsSum<double, double>(double *A, double *B, double *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);


/**
* calcs A*coefficents_set->A_coefficent+B*coefficents_set->B_coefficent => C for calculating shigh and ac+dc summary created with consistent indexes calculations
*
*/
template<typename T1, typename T2> __global__ void CudaVectorsSumNSquare(T1 *A, T2 *B, T2 *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set)
{
	lambdaFloat coefficentA = lambdaFloat(coefficents_set->A_coefficent);
	lambdaFloat coefficentB = lambdaFloat(coefficents_set->B_coefficent);
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	if (coefficents_set->reverseCoefficents) {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSumDivide(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaVectorSumDivideNSquare(A, B, C, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}
	else {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSum(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaVectorSumNSquare(A, B, C, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}

	__syncthreads();
}



// cuda vector sum
template __global__ void CudaVectorsSumNSquare<float, double>(float *A, double *B, double *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);
template __global__ void CudaVectorsSumNSquare<float, float>(float *A, float *B, float *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);
template __global__ void CudaVectorsSumNSquare<double, double>(double *A, double *B, double *C, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);


/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p src is global input array
* \p target is global output array
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<class T> __device__ void DeviceCudaSquare(T *src, T *target, int offset, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster)
{
	int k;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster){
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		target[current_offset] = src[current_offset] * src[current_offset];
	}
}


template __device__ void DeviceCudaSquare<lambdaFloat>(lambdaFloat *src, lambdaFloat *target, int offset, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster);


/**
* calcs src.*src => target for Shigh.^2=>dS summary created with consistent indexes calculations
*
*/
template<class T> __global__ void CudaSquare(T *src, T *target, device_params *filter_params)
{
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	DeviceCudaSquare<T>(src,target, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	__syncthreads();
}



// vector square
template __global__ void CudaSquare<lambdaFloat>(lambdaFloat *src, lambdaFloat *target, device_params *filter_params);



/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p IHC is global ihc(both input and output) array
* \p IHC_Damage_Factor is calculated ihc health actually such that 10^8 is healthy and 0 is completely lost
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T,typename T2> __device__ void DeviceCudaCalcIHC(T *PRE_IHC, T2 *IHC, double IHC_Damage_Factor, int offset, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster)
{
	int k;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster){
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		IHC[current_offset] = T2(log10(fmax(double(IHC_Damage_Factor*PRE_IHC[current_offset]), 0.0) + EPS));
	}
}

/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p IHC is global ihc(both input and output) array
* \p IHC_Damage_Factor is calculated ihc health actually such that 10^8 is healthy and 0 is completely lost
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T, typename T2> __device__ void DeviceCudaCalcIHCComposite(T *AC_response,T *DC_response, T2 *IHC, double IHC_Damage_Factor, int offset, int time_length_analysis, int cochlea_sections, double coefficent_AC, double coefficent_DC, int time_node_offset, int time_node_jumps_in_cluster)
{
	int k;
	int current_offset;
	double PRE_IHC;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		PRE_IHC = fmaf(AC_response[current_offset], coefficent_AC, DC_response[current_offset] * coefficent_DC);
		IHC[current_offset] = T2(log10(fmax(double(IHC_Damage_Factor*PRE_IHC), 0.0) + EPS));
	}
}

/**
* device run for single time block of single index	of vector summary
* note offset here is the offset for start of time node name
* \p IHC is global ihc(both input and output) array
* \p IHC_Damage_Factor is calculated ihc health actually such that 10^8 is healthy and 0 is completely lost
* \p offset index of start calculation in output array, its already considarate the spatial section on the cochlea
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T, typename T2> __device__ void DeviceCudaCalcIHCCompositeDivide(T *AC_response, T *DC_response, T2 *IHC, double IHC_Damage_Factor, int offset, int time_length_analysis, int cochlea_sections, double coefficent_AC, double coefficent_DC, int time_node_offset, int time_node_jumps_in_cluster)
{
	int k;
	int current_offset;
	double PRE_IHC;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		PRE_IHC = (AC_response[current_offset]/ coefficent_AC)+(DC_response[current_offset] / coefficent_DC);
		IHC[current_offset] = T2(log10(fmax(double(IHC_Damage_Factor*PRE_IHC), 0.0) + EPS));
	}
}




/**
* calcs lg10(max(IHC,0)*IHC_Damage_vector+EPS) into IHC vector
* this procedure will run after AC*eta_AC+DC*eta_DC
*
*/
template<typename T, typename T2> __global__ void CudaCalcIHC(T *PRE_IHC, T2 *IHC, device_params *filter_params)
{
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	double IHC_Damage_Factor = CUDA_IHC_DAMAGE[cochlea_offset_section];
	DeviceCudaCalcIHC<T,T2>(PRE_IHC,IHC, IHC_Damage_Factor, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	__syncthreads();
}

/**
* calcs lg10(max(IHC,0)*IHC_Damage_vector+EPS) into IHC vector
* this procedure will run after AC*eta_AC+DC*eta_DC
*
*/
template<typename T, typename T2> __global__ void CudaCalcIHCComposite(T *AC_response, T *DC_response, T2 *IHC, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set)
{
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
	T coefficentAC = T(coefficents_set[1].A_coefficent);
	T coefficentDC = T(coefficents_set[1].B_coefficent);
	int reverseCoefficents = coefficents_set[1].reverseCoefficents;																																  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	double IHC_Damage_Factor = CUDA_IHC_DAMAGE[cochlea_offset_section];
	if (reverseCoefficents) {
		DeviceCudaCalcIHCCompositeDivide<T, T2>(AC_response, DC_response, IHC, IHC_Damage_Factor, offset, time_length_analysis, cochlea_sections, double(coefficentAC), double(coefficentDC), time_node_offset, time_node_jumps_in_cluster);
	}
	else {
		DeviceCudaCalcIHCComposite<T, T2>(AC_response, DC_response, IHC, IHC_Damage_Factor, offset, time_length_analysis, cochlea_sections, double(coefficentAC), double(coefficentDC), time_node_offset, time_node_jumps_in_cluster);
	}

	
	__syncthreads();
}



// calc ihc isolated to templates
template __global__ void CudaCalcIHC<lambdaFloat,float>(lambdaFloat *PRE_IHC, float *IHC, device_params *filter_params);

// calc ihc isolated to templates
template __global__ void CudaCalcIHCComposite<lambdaFloat, float>(lambdaFloat *AC_response, lambdaFloat *DC_response, float *IHC, device_params *filter_params, vectors_sum_linear_coefficents *coefficents_set);


/**
* device run for single time block of single index	of lambda calculation
* note offset here is the offset for start of time node name
* \p IHC is global ihc, input array
* \p Lambda is global output array
* \p cochlea_offset_section index of start calculation in output array, its the spatial section on the cochlea	+ lambda block index offset
* \p time_length_analysis number of indexes in output array to calculate
* \p lambda_index is the id of lambda block and spont rate index in unified nerves parameter  
* \p A_index is the id A to take value in index in unified nerves parameter
* \p time_length_analysis number of indexes in output array to calculate
* \p cochlea_sections is number of spatial cochlea sections
* \p time_node_offset is number of time nodes from the start offset the algorithm will calculate
* \p time_node_jumps_in_cluster time nodes jump between each consecutive calculation
*/
template<typename T1,typename T2> __device__ void DeviceCudaCalcLambda(T1 *IHC, float *Lambda, T2 *JND_Lambda, int cochlea_offset_section, int lambda_write_offset, int lambda_index, int A_index, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster, float Lambda_SAT)
{
	int k = 0;
	int current_offset; // progress on the input since we jump in sections every time
	int write_offset;
	double base_lambda;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster){
		current_offset = cochlea_offset_section + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		write_offset = current_offset + lambda_write_offset;
		base_lambda = fmin(double(Lambda_SAT - CUDA_Nerves_Clusters[lambda_index]), fmax(double(CUDA_Nerves_Clusters[A_index]) * IHC[current_offset], 0.0));
		JND_Lambda[write_offset] = base_lambda;
		Lambda[write_offset] = float(base_lambda) + CUDA_Nerves_Clusters[lambda_index];
		//if ( current_offset>=350000&&current_offset<=350256 ) printf("Y[%d]==%.3e,time_length_analysis=%d\n",current_offset,Y[current_offset],time_length_analysis);
	}
}

// device calc ihc isolated for templates
template __device__ void DeviceCudaCalcLambda<lambdaFloat,double>(lambdaFloat *IHC, float *Lambda, double *JND_Lambda, int cochlea_offset_section, int lambda_write_offset, int lambda_index, int A_index, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster, float Lambda_SAT);

/*, float *Lambda*/
/**
*	calculationg the lambda blocks from the IHC array by min(RSAT,SpontRate[lambda_type]+max(A[lambda_type]*IHC,0))
*/
template<typename T1,typename T2> __global__ void CudaCalcLambda(T1 *IHC,T2 *Lambda_Buffer, T2 *JND_Lambda, device_params *filter_params,int save_lambda)
{
	// in this procedure lambda offset is ignored
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int intervalsNum = totalDecoupledBlocks(gridDim) / LAMBDA_COUNT;
	int lambda_index = decoupledBlockId(blockIdx, gridDim) / intervalsNum;
	int interval_id = decoupledBlockId(blockIdx, gridDim) - lambda_index*intervalsNum;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	
	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(interval_id*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	//int A_index = lambda_index + LAMBDA_COUNT;
	int lambda_write_offset = lambda_index*filter_params->calcTime*cochlea_sections;
	
	double zero_factor = cochlea_offset_section != 0 ? 1.0 : 0.0;
	
	int write_offset;
	int k;
	int current_offset;
	double base_lambda;
	double Lambda_SAT = filter_params->Lambda_SAT;
	double Aihc = double(model_Aihc[lambda_index*SECTIONS + cochlea_offset_section]);
	double spont = double(CUDA_Nerves_Clusters[lambda_index]);
	double zero_offset = cochlea_offset_section != 0 ? 0.0 : spont;
	//DeviceCudaCalcLambda(IHC, Lambda, JND_Lambda, cochlea_offset_section, lambda_write_offset, lambda_index, A_index, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster, filter_params->Lambda_SAT);
	for (k = time_node_offset; k<time_length_analysis; k++) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		write_offset = current_offset + lambda_write_offset;
		base_lambda = fmin(Lambda_SAT, fmax(Aihc*double(IHC[current_offset]),spont));
		JND_Lambda[write_offset] = T2(fma(zero_factor,base_lambda,zero_offset));
		if (save_lambda) {
			Lambda_Buffer[write_offset] = T2(fma(zero_factor, base_lambda, zero_offset));
		}
		//if ( current_offset>=350000&&current_offset<=350256 ) printf("Y[%d]==%.3e,time_length_analysis=%d\n",current_offset,Y[current_offset],time_length_analysis);
	}
	
	__syncthreads();
}

// calc lambdas types of decalarations
template __global__ void CudaCalcLambda<lambdaFloat, double>(lambdaFloat *IHC, double *Lambda_Buffer, double *JND_Lambda, device_params *filter_params,int save_lambda);

template<typename T1, typename T2> __device__ void DeviceCopy_Array(volatile T1 *src, volatile T2 *dst, int offset, int time_length_analysis, int cochlea_sections, int time_node_offset, int time_node_jumps_in_cluster) {
	int k = 0;
	int current_offset;
	for (k = time_node_offset; k<time_length_analysis; k += time_node_jumps_in_cluster) {
		current_offset = offset + k*cochlea_sections; // untransposed adding sections multiplication for k, offset time
		dst[current_offset] = src[current_offset] ;
	}
}

/**
*   inside equtaions references to Yonatan Koral Efficent Tool Thesis
*	this global cuda procedure calculate the IHC array from the Basialr Membrane array results using device functions
*	calculate the AC
*	calculate the SHigh
*	calculate the dS
*	calculate the DC
*	calculate the IHC
*/
template<typename T> __global__ void CudaUnifiedCalcIHC(
	float *BM_internal,
	T *cuda_Buffer2,
	T *cuda_Buffer3,
	T *cuda_Buffer4,
	T *cuda_BufferOutput,
	device_params *filter_params,
	vectors_sum_linear_coefficents *coefficents_set,
	int backup_stage) {
	// first filter paramets for ac filter from filter params index 0
	int filter_index = filter_params->filter_b_start_index + 1;
	int filter_size = filter_params->order_of_ac_filter; //(int)(device_time_filter[filter_index]+0.1f); // very important filter data start from index 1 0 index is size....
	int cochlea_offset_section = calcCochleaSection(blockIdx, blockDim, threadIdx);
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->calcTime - lambda_offset; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = calcTimeNodesJump(blockIdx, blockDim, threadIdx); // block dim y represent num of threads per sections so each thread will jump by this number times the number of sections
	int offset = calcStartMainOffset(blockIdx, blockDim, threadIdx, lambda_offset, cochlea_sections);
	int time_node_offset = calcTimeNodesOffset(blockIdx, blockDim, threadIdx);
	// first sum is bm internal and cuda AC uses coefficents set from index 0
	T coefficentA = T(coefficents_set->A_coefficent);
	T coefficentB = T(coefficents_set->B_coefficent);
	
	double IHC_Damage_Factor = CUDA_IHC_DAMAGE[cochlea_offset_section];
	// calculate unified block to include decoupling, note value is same on both fir filters
	int calculatedIntervalTimeNodes = filter_params->FilterDecoupledMode ? filter_params->intervalTimeNodes*filter_params->Decouple_Filter : time_length_analysis;
	int final_regular_division_position = filter_params->intervalTimeNodes*filter_params->time_blocks;
	// ac will only be calculated if its not IIR filter otherwise it will be calculated seperatly - calculating Eq. 5.1
	if (filter_params->filter_a_start_index == -1) {
		// first stage calculate the AC
		DeviceCudaFIRFilter<float, T>(BM_internal, cuda_Buffer2, offset, time_length_analysis, cochlea_sections, filter_size, filter_index, time_node_offset, time_node_jumps_in_cluster, calculatedIntervalTimeNodes, final_regular_division_position);
	}
	if (backup_stage == 9) {  // AC backup
		DeviceCopy_Array<JNDFloat, JNDFloat>(cuda_Buffer2, cuda_BufferOutput, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	}
	__syncthreads();
	// calculating the summary of bm internal and ac  each thread handles only its own result of ac so synchronization unecessary
	// calculating Eq. 5.2
	DeviceCudaVectorSumNSquare<float, T>(BM_internal, cuda_Buffer2, cuda_Buffer3, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	__syncthreads();

	if (backup_stage == 12 || backup_stage == 11) {  // SHigh backup
		DeviceCopy_Array<JNDFloat, JNDFloat>(cuda_Buffer3, cuda_BufferOutput, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	}

	// updating filter parameters for the dc filter now its from index 1, pre filter synchronization is necessary to ensure DS array is valid
	filter_index = filter_params[1].filter_b_start_index+1;
	filter_size = filter_params->order_of_dc_filter; // dc filter size
	// calculating the DC filter - Eq 5.3
	DeviceCudaFIRFilter<T, T>(cuda_Buffer3, cuda_Buffer4, offset, time_length_analysis, cochlea_sections, filter_size, filter_index, time_node_offset, time_node_jumps_in_cluster, calculatedIntervalTimeNodes, final_regular_division_position);
	__syncthreads();
	if (backup_stage == 10) {	 // DC backup
		DeviceCopy_Array<JNDFloat, JNDFloat>(cuda_Buffer4, cuda_BufferOutput, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	}
	// now setting the coefficents for pre IHC calculator
	coefficentA =T(coefficents_set[1].A_coefficent);
	coefficentB =T(coefficents_set[1].B_coefficent);
	int reverseCoefficents = coefficents_set[1].reverseCoefficents;
	// calculating the AC and DC for pre IHC, AC is already valid and since each thread use only its own results so synchronization is unecessary
	// calculating equations 5.4 - 5.5, this was tested with divided parmeters and multiplied parmeters due to previous calculation error on my part (which was fixed), you can use either
	if (reverseCoefficents) {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSumDivide(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaCalcIHCCompositeDivide<T, JNDFloat>(cuda_Buffer2, cuda_Buffer4, BM_internal, IHC_Damage_Factor, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}
	else {
		//if (threadIdx.x == 0) printf("DeviceCudaVectorSum(A,B,C,%d,%d,%d,%.2f,%.2f,%d,%d)\n", offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
		DeviceCudaCalcIHCComposite<T, JNDFloat>(cuda_Buffer2, cuda_Buffer4, BM_internal, IHC_Damage_Factor, offset, time_length_analysis, cochlea_sections, coefficentA, coefficentB, time_node_offset, time_node_jumps_in_cluster);
	}
	if (backup_stage == 14 || backup_stage==13) { // PRE IHC backup
		DeviceCopy_Array<JNDFloat, JNDFloat>(BM_internal, cuda_BufferOutput, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
	}



	__syncthreads();
}


// possible types for iir on device

// specializing calculation of IHC
template __global__ void CudaUnifiedCalcIHC<JNDFloat>(
	float *BM_internal,
	JNDFloat *cuda_Buffer2,
	JNDFloat *cuda_Buffer3,
	JNDFloat *cuda_Buffer4,
	JNDFloat *cuda_BufferOutput,
	device_params *filter_params,
	vectors_sum_linear_coefficents *coefficents_set,
	int backup_stage);

template<typename T1, typename T2> __global__ void copyDeviceBackup(T1 *src, T2 *cudaBackupArray, device_params *filter_params) {
	int cochlea_offset_section = threadIdx.x; // 
	int lambda_offset = filter_params->lambda_offset;
	int time_length_analysis = filter_params->intervalTimeNodes / gridDim.z; // here I run on the entire set
	int cochlea_sections = filter_params->cochlea_sections; // number of cochlea space sections
	int time_node_jumps_in_cluster = 1;

	/*
	*  start offset represents calculated section + lambda offset_in_time*sections
	*/
	int offset = cochlea_offset_section + cochlea_sections*(decoupledBlockId(blockIdx, gridDim)*time_length_analysis + lambda_offset);//calcStartMainOffset(blockIdx,blockDim,threadIdx,lambda_offset,cochlea_sections);
																																	  // each thread start from its own adjusted offset in the time block offset
	int time_node_offset = 0;//calcTimeNodesOffset(blockIdx,blockDim,threadIdx);
	DeviceCopy_Array<T1, T2>(src, cudaBackupArray, offset, time_length_analysis, cochlea_sections, time_node_offset, time_node_jumps_in_cluster);
}


// calc ihc isolated to templates
template __global__ void copyDeviceBackup<float, JNDFloat>(float *src, JNDFloat *cudaBackupArray, device_params *filter_params);
//template __global__ void copyDeviceBackup<JNDFloat, JNDFloat>(JNDFloat *src, JNDFloat *cudaBackupArray, device_params *filter_params);

void runIIRKernelByParams(int Show_Run_Time,Log &outer_log) {
	dim3 filtersIIRThreads(SECTIONS, 1, 1);
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 16);
	cudaEventsStartTimer(start, stop, Show_Run_Time & 16);
	if (cudaJNDHolder.host_local_param[0].FilterDecoupledMode) {
		int dfilter = cudaJNDHolder.host_local_param[0].Decouple_Filter > 0 ? cudaJNDHolder.host_local_param[0].Decouple_Filter : 1;
		dim3 decoupledGrid(cudaJNDHolder.host_local_param[0].time_blocks / dfilter, 1, 1);
		//std::cout << "CudaIIRFilterDecoupled<float, lambdaFloat> << <" << showDIM3(decoupledGrid) << ", " << showDIM3(filtersIIRThreads) << " >> >(cudaHolderData.cuda_saved_speeds, cuda_JND_Lambda, global_device_params);" << std::endl;
		CudaIIRFilterDecoupled<float, lambdaFloat> KERNEL_ARGS2(decoupledGrid, filtersIIRThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_JND_Lambda, cudaJNDHolder.global_device_params);
	}
	else {
		// for IIR will parrallel in space only, not time so only single block with SECTIONS threads
		dim3 filtersIIRGrid(1, 1, 1);
		CudaIIRFilter<float, lambdaFloat> KERNEL_ARGS2(filtersIIRGrid, filtersIIRThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_JND_Lambda, cudaJNDHolder.global_device_params);
	}
	
	outer_log.timeAtFlag(40, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 16, "IIR calculation time"), Show_Run_Time & 16);
}

void setDecoupledRun(dim3 &filtersGrid, dim3 &filtersThreads, const int& intervals_num, const int&blocks_per_interval, const int& Decouple_Unified_IHC_Factor) {
	filtersGrid.x = intervals_num;
	filtersGrid.y = blocks_per_interval; // decoupled blocks+ extra decoupling for better division
	filtersGrid.z = Decouple_Unified_IHC_Factor>0? Decouple_Unified_IHC_Factor:1;
	filtersThreads.x = SECTIONS;
	filtersThreads.y = 1;
	filtersThreads.z = 1;

}
// calculating IHC by stages Cochlear Model for Hearing Loss, equations numbers from Yonatan Koral Thesis, Efficent Tool For Cochlea Simulation
extern "C" void RunIHCKernel(JNDFloat *host_backup, int Show_Run_Time, int save_lambda, int backup_stage,int Decouple_Unified_IHC_Factor,Log &outer_log) {
	// if the data loaded from hd its on host and it needs to be first time or its not relevant
	
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 8);
	dim3 filtersGrid(IHC_FILTER_BLOCK, 1, 1);
	dim3 filtersThreads(SECTIONS_PER_IHC_FILTER_BLOCK, THREADS_PER_IHC_FILTER_SECTION, 1);

	int lambda_write_offset = cudaJNDHolder.host_local_param->calcTime*SECTIONS;
	// copy from  saved speeds the rest of the data
	
	// unfied calculation of the ihc
	cudaEventsStartTimer(start, stop, Show_Run_Time & 8);
	if (Decouple_Unified_IHC_Factor<=0) {
		if (cudaJNDHolder.host_local_param[0].order_of_ac_filter > -1) {
			// calculating AC stage if AC filter is IIR, Eq. 5.1
			
			runIIRKernelByParams(Show_Run_Time,outer_log);
		}
		// rest of the IHC process calculated in single kernel - Eq. 5.2 - 5.5
		CudaUnifiedCalcIHC<JNDFloat> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_JND_Lambda, cudaLambdaHolderData.cuda_JND_Lambda + lambda_write_offset, cudaLambdaHolderData.cuda_JND_Lambda + 2*lambda_write_offset, cudaLambdaHolderData.cuda_Buffer1, cudaJNDHolder.global_device_params, cudaJNDHolder.vectors_sums_coefficents,backup_stage);
	} else {
		// this is effectively cause the grid to be one large interval if decoupler size is 0		
		int dfilter = cudaJNDHolder.host_local_param[0].Decouple_Filter > 0 ? cudaJNDHolder.host_local_param[0].Decouple_Filter : cudaJNDHolder.host_local_param[0].time_blocks;
		setDecoupledRun(filtersGrid,filtersThreads, cudaJNDHolder.host_local_param[0].time_blocks / dfilter, dfilter, Decouple_Unified_IHC_Factor);
		if (cudaJNDHolder.host_local_param[0].order_of_ac_filter == -1) {
			// calculating the lambda by multiple kernel currently remain for backward reference
			// ac filter run, FIR filter Eq. 5.1
			CudaFIRFilter<float, lambdaFloat> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_JND_Lambda, cudaJNDHolder.global_device_params);
		}
		else {
			// ac a filter make IIR present, Eq. 5.1
			runIIRKernelByParams(Show_Run_Time,outer_log);
		}
		if (backup_stage == 9) {  // AC backup
			copyDeviceBackup<JNDFloat, JNDFloat>KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaLambdaHolderData.cuda_JND_Lambda, cudaLambdaHolderData.cuda_Buffer1, cudaJNDHolder.global_device_params);
		}
		// calcs Eq 5.2 - dS
		CudaVectorsSumNSquare<float, lambdaFloat> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_JND_Lambda, cudaLambdaHolderData.cuda_JND_Lambda + lambda_write_offset, cudaJNDHolder.global_device_params, &cudaJNDHolder.vectors_sums_coefficents[0]);
		if (backup_stage == 12 || backup_stage == 11) {  // SHigh backup
			copyDeviceBackup<JNDFloat, JNDFloat>KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaLambdaHolderData.cuda_JND_Lambda + lambda_write_offset, cudaLambdaHolderData.cuda_Buffer1, cudaJNDHolder.global_device_params);
		}
		// caculating Eq 5.3 - DC response
		CudaFIRFilter<lambdaFloat, lambdaFloat> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaLambdaHolderData.cuda_JND_Lambda + lambda_write_offset, cudaLambdaHolderData.cuda_JND_Lambda + 2*lambda_write_offset, &cudaJNDHolder.global_device_params[1]);
		if (backup_stage == 10) {  // DC backup
			copyDeviceBackup<JNDFloat, JNDFloat>KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaLambdaHolderData.cuda_JND_Lambda + 2*lambda_write_offset, cudaLambdaHolderData.cuda_Buffer1, cudaJNDHolder.global_device_params);
		}
		
		// calculates Eq 5.4 and 5.5 - Log of Psi IHC
		CudaCalcIHCComposite<lambdaFloat, float> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaLambdaHolderData.cuda_JND_Lambda, cudaLambdaHolderData.cuda_JND_Lambda + 2 * lambda_write_offset, cudaHolderData.cuda_saved_speeds, cudaJNDHolder.global_device_params, &cudaJNDHolder.vectors_sums_coefficents[1]);
		// calc the IHC log
		if (backup_stage == 13 || backup_stage == 14) {  // IHC backup
			copyDeviceBackup<JNDFloat, JNDFloat>KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_Buffer1, cudaJNDHolder.global_device_params);
		}
	
	}
	if (backup_stage >= 9 && backup_stage <= 14) {
		GeneralKernel_Copy_Results_Template<JNDFloat>(host_backup, cudaLambdaHolderData.cuda_Buffer1, lambda_write_offset);
	}
	
	outer_log.timeAtFlag(41, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 8, "IHC calculation time"), Show_Run_Time & 8);
	/// calcs the lambda itself
	
	cudaEventsStartTimer(start, stop, Show_Run_Time & 8);
	/* cuda_Lambda,*/
	int dfilter = cudaJNDHolder.host_local_param[0].Decouple_Filter > 0 ? cudaJNDHolder.host_local_param[0].Decouple_Filter : cudaJNDHolder.host_local_param[0].time_blocks;
	setDecoupledRun(filtersGrid, filtersThreads, LAMBDA_COUNT* cudaJNDHolder.host_local_param[0].time_blocks / dfilter, dfilter, Decouple_Unified_IHC_Factor);// changed grid to support all lambda calculations
	// Calculates the AN response for all groups of Neurons Eq 5.6 - 5.8
	CudaCalcLambda<lambdaFloat, JNDFloat> KERNEL_ARGS2(filtersGrid, filtersThreads)(cudaHolderData.cuda_saved_speeds, cudaLambdaHolderData.cuda_Buffer1, cudaLambdaHolderData.cuda_JND_Lambda, cudaJNDHolder.global_device_params, save_lambda);
	
	outer_log.timeAtFlag(42, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 8, "Lambda calculation time"), Show_Run_Time & 8);
}


/**
* src is __global__ array in cuda device
* target is array in host
* size is array length
* function will copy from src to taraget use cudaMemcpy
*/
extern "C" void GeneralKernel_Copy_Results(float *target,float *src, size_t size) noexcept(false) {
	const size_t sizer = size*sizeof(float);
	//printf("copy %d bytes to host\n",sizer);
	gpuAssert(cudaMemcpy((void *)target,src,sizer,cudaMemcpyDeviceToHost));
}


/**
* src is __global__ array in cuda device
* target is array in host
* size is array length
* function will copy from src to taraget use cudaMemcpy
*/
extern "C" void GeneralKernel_Copy_Results_Double(double *target, double *src, size_t size) noexcept(false) {
	const size_t sizer = size*sizeof(double);
	//printf("copy %d bytes to host\n",sizer);
	gpuAssert(cudaMemcpy((void *)target, src, sizer, cudaMemcpyDeviceToHost));
}

/**
* src is __global__ array in cuda device
* target is array in host
* size is array length
* function will copy from src to taraget use cudaMemcpy
*/
template<class T> void GeneralKernel_Copy_Results_Template(T *target, T *src, size_t size, size_t offset) {
	const size_t sizer = size*sizeof(T);
	
//	printf("GeneralKernel_Copy_Results_Template copying %d with offset %d,number of bytes: %d\n,Stack Trace: %s", size, offset,sizer,oss.str().c_str());
	gpuAssert(cudaMemcpy((void *)target, src+offset, sizer, cudaMemcpyDeviceToHost));
}

template<class T> void GeneralKernel_Copy_Results_Template(T *target, T *src, size_t size) {
	GeneralKernel_Copy_Results_Template<T>(target, src, size, 0);
}



template<class T> void ReverseKernel_Copy_Results_Template(T *cpu_src, T *cuda_target, size_t start_time_node, size_t time_nodes, int sections) {
	gpuAssert(cudaMemcpy((void *)cuda_target, &cpu_src[start_time_node*sections], time_nodes*sections*sizeof(T), cudaMemcpyHostToDevice));
}


/**
* device run for calculation of accumulation of length cells at jump_size interval between them
* \p src is input summed array
* \p dst is pointer to target result
* \p jump_size  is interval between to accumulated cells
* \p length number of accumulated cells
*/
__device__ void CalcAccumulation(float *src,float *dst,int jump_size,int length) {
	float accu = 0.0f;
	for (int index = 0; index < length*jump_size; index += jump_size) {
		accu += src[index];
	}
	*dst = accu;
}

/**
* device run for calculation of average of length cells at jump_size interval between them
* \p src is input summed array
* \p dst is pointer to target result
* \p jump_size  is interval between to accumulated cells
* \p length number of accumulated cells
*/
__device__ void CalcAvg(float *src, float *dst, int jump_size, int length) {
	CalcAccumulation(src, dst, jump_size, length);
	*dst = *dst / float(length);
}

/*
* calculates dA from input will run in single block with #threads as number of intervals
*/
__global__ void cudaCalculateDA(float *input, device_jnd_params *dA, int JNDIntervalNodes, int JNDIntervalHeadNodes, int JNDIntervalActualNodes, int offset_start) {
	float acc = 0.0f;
	float current = 0.0f;
	int start_index = offset_start + threadIdx.x*JNDIntervalNodes + JNDIntervalHeadNodes;
	int end_index = start_index + JNDIntervalActualNodes;
	for (int index = start_index; index < end_index; index++) {
		current = input[index];
		current = current*current;
		if (acc < current) acc = current;
	}
	dA[threadIdx.x].dA = sqrtf(acc);
	__syncthreads();
}

// this function calculates average of lambda (part of Eq.17) due to synchronization issues
__global__ void GlobalCalculateMeanRate(
	device_jnd_params *dA,
	JNDFloat *JND_Lambda,
	JNDFloat *MeanRate,
	int lengthOffset,
	int calculatedMatrixSize,
	int JNDIntervals, // local	
	int JNDIntervalLength,
	int JNDIntervalHeadNodes,
	int overlapNodes,
	int JND_USE_Spont_Base_Reference_Lambda
	){
	int lambda_index = blockIdx.y;
	int dAindex = blockIdx.x;
	int section_index = threadIdx.x;
	int sections = blockDim.x;
	int avg_fisher_full_index = lambda_index*JNDIntervals + dAindex;
	int block_offset = lambda_index*calculatedMatrixSize + section_index + sections*(dAindex*JNDIntervalLength + JNDIntervalHeadNodes);
	int mean_rate_offset = avg_fisher_full_index*sections + section_index;
	double meanRateAccumulator = JND_Lambda[block_offset];
	// averaging work per thread no need to loops
	// will calculate manually to improve precisions
	for (int time_offset = 1; time_offset < lengthOffset; time_offset++) {
		//meanRateDiff = JND_Lambda[block_offset + sections*time_offset];// -lambdaBase;
		meanRateAccumulator = meanRateAccumulator + JND_Lambda[block_offset + sections*time_offset];
	}

	MeanRate[mean_rate_offset] = JNDFloat(meanRateAccumulator) / JNDFloat(lengthOffset);
	__syncthreads();

}
// uses the result of GlobalCalculateMeanRate to calculate on GPU eq 17-20)
__global__ void GlobalCalculateJND(
	bool calculate_ai,
	bool calculate_rms,
	device_jnd_params *dA,
	JNDFloat *JND_Lambda,
	JNDFloat *MeanRate,
	JNDFloat *Buffer1,
	double *nIHC,
	double Fs,
	double scaleBMVelocityForLambdaCalculation,
	int writeMatrixSize,
	int calculatedMatrixSize,
	int overlapNodes,
	int JNDIntervalHeadNodes,
	int JNDIntervalLength,
	int lengthOffset,
	int JNDIntervals, // local	   
	int JNDIntervalsFull, // local
	int *JND_Calculated_Intervals,  // global
	int numOFJNDCalculated,	// global
	int *JND_Refrence_Intervals, // global
	int numOFJNDReferences,	 // global
	int handeledIntervalsJND, // already handeled intervals
	int *JND_Serial_Intervals_Positions,
	int *JND_Interval_To_Reference,
	JNDFloat *F_RA, // result for fisher rate not lambda summed, but time and space reduced
	JNDFloat *FisherAISum, // result for fisher AI not lambda summed, but time and space reduced	 
	double JND_Delta_Alpha_Length_Factor,
	device_params *general_params,
	int isdACalced,
	int JND_USE_Spont_Base_Reference_Lambda,
	int backup_stage
	) {

	__shared__ JNDFloat shared_acc_rms[SECTIONS];
	__shared__ JNDFloat shared_acc_ai[SECTIONS];


	 int lambda_index = blockIdx.y;
	 int dAindex = blockIdx.x;
	 int section_index = threadIdx.x;
	 int sections = blockDim.x;
	 int avg_fisher_full_index = lambda_index*JNDIntervals + dAindex;
	 int mean_rate_offset = avg_fisher_full_index*sections + section_index;
	 JNDFloat T = JND_Delta_Alpha_Length_Factor / Fs;
	 JNDFloat Tlength = float(lengthOffset) / Fs;
	 //JNDFloat lambdaBase = CUDA_Nerves_Clusters[lambda_index];
	int globaldAIndex = dAindex + handeledIntervalsJND;
	// special control mechanism for mean rate calculation that avrages mean rate (not dMeanRate)
	// to ensure that summary is larger enough than reference (so we are not just square negative values that will create artifacts)
	if (calculate_rms) {
		// test RMS average for debug output
		shared_acc_rms[section_index] = MeanRate[mean_rate_offset];
		// reducing spatial dimension
		for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
			__syncthreads();
			if (section_index<t_i) {
				shared_acc_rms[section_index] = shared_acc_rms[section_index] + shared_acc_rms[section_index + t_i];
			}

		}
		__syncthreads();
		if (section_index == 0 && backup_stage==1) {
			
			Buffer1[avg_fisher_full_index] = shared_acc_rms[section_index] / (JNDFloat(sections));
		}
		

	}
	__syncthreads();

	int globalReferenceInterval = globaldAIndex;
	bool isRefrence = numOFJNDReferences > 0;
	// this means we have actually refrences to test
	if (isRefrence) {
		isRefrence = false; // now for the actual test
		for (int index = 0; index < numOFJNDReferences; index++) {
			if (JND_Refrence_Intervals[index] == globaldAIndex) {
				isRefrence = true;
				break;
			}
		}
		

	}
	__syncthreads();
	// find for each calculated JND signal+noise block its pure Noise block
	if (!isRefrence) {
		// assuming everything has reference
		globalReferenceInterval = JND_Interval_To_Reference[JND_Serial_Intervals_Positions[globalReferenceInterval]];
	}
	__syncthreads();
	int dAreferenceIndex = globalReferenceInterval - handeledIntervalsJND; // to find local index on tested output
	
	int mean_rate_reference_offset = (avg_fisher_full_index + dAreferenceIndex - dAindex)*sections + section_index;
	JNDFloat dAvalue = dA[isdACalced*dAindex + (1 - isdACalced)*globaldAIndex].dA;
	JNDFloat dMRate = (MeanRate[mean_rate_offset] - MeanRate[mean_rate_reference_offset]) / dAvalue;
	if (backup_stage == 2) {
		Buffer1[mean_rate_offset] = dMRate;
	}
	if (calculate_ai) {
		int calculate_lambda_offset = lambda_index*calculatedMatrixSize + section_index + sections*(dAindex*JNDIntervalLength + JNDIntervalHeadNodes);
		int reference_lambda_offset = lambda_index*calculatedMatrixSize + section_index + sections*(dAreferenceIndex*JNDIntervalLength + JNDIntervalHeadNodes);
		JNDFloat preFisherAITimeReducedValue = 0.0;
		for (int time_offset = 0; time_offset < lengthOffset; time_offset++) {
			JNDFloat refLambda = JND_Lambda[reference_lambda_offset];
			JNDFloat calcedLambda = JND_Lambda[calculate_lambda_offset];
			JNDFloat dLambdaCalculated = dAvalue > 0 ? (calcedLambda - refLambda) / dAvalue : 0;
			if (backup_stage == 3) {
				Buffer1[calculate_lambda_offset] = dLambdaCalculated;
			}
			/*
			* calculating pre fisher AI
			* from matlab
			* fisher AI : Ts*(dL.^2./reshape(RefLamda(j,:,:),Nsec,Time)	=> into pre fisher AI
			* VERY Important Correction: original division from matlab program incorrect:
			* JNDFloat preFisherAIValue = refLambda>0 ? (dLambdaCalculated*dLambdaCalculated / refLambda / Fs) : 0;
			* since its contradict eq 19 in miriam's article
			*/
			JNDFloat preFisherAIValue = (dLambdaCalculated*dLambdaCalculated / (Fs*refLambda)) ;
			if (backup_stage == 4) {
				Buffer1[calculate_lambda_offset] = preFisherAIValue;
			}
			preFisherAITimeReducedValue += preFisherAIValue;
			reference_lambda_offset += sections;
			calculate_lambda_offset += sections;
		}
		preFisherAITimeReducedValue = rsqrt(preFisherAITimeReducedValue);
		JNDFloat preFisherAIValue = (T/Tlength)*nIHC[section_index] /(preFisherAITimeReducedValue* preFisherAITimeReducedValue);
		if (backup_stage == 5) {
			Buffer1[mean_rate_offset] = preFisherAIValue;
		}
		shared_acc_ai[section_index] = preFisherAIValue;
		
	}

	/*
	* calculate pre fisher values before summering
	* from matlab
	* fisher rate : nIHC.*Tmean./RefMeanRate(j,:).*(dMeanRate(j,:).^2)	 => into pre fisher rate
	*/
	if (calculate_rms) {
		JNDFloat MeanRateReferenced = MeanRate[mean_rate_reference_offset]; 
		// all mean rates are actually multiplied by lambda base, so no needd to multiply by length offset on nominator
		JNDFloat CRLB_RAValue = rsqrt(T / MeanRateReferenced*dMRate * dMRate);
		if (backup_stage == 6) {
			Buffer1[mean_rate_offset] = CRLB_RAValue;
		}
		shared_acc_rms[section_index] = nIHC[section_index] /(CRLB_RAValue*CRLB_RAValue);
	}
	__syncthreads();
	 // reducing spatial dimension for AI/RMS
	for (int t_i = (SECTIONS >> 1); t_i >= 1; t_i >>= 1) {
		__syncthreads();
		if (section_index<t_i) {
			if (calculate_ai) shared_acc_ai[section_index] = shared_acc_ai[section_index] + shared_acc_ai[section_index + t_i];

			if (calculate_rms) shared_acc_rms[section_index] = shared_acc_rms[section_index] + shared_acc_rms[section_index + t_i];
		}

	}

	__syncthreads();
	// calculating fisher number for each block on AI/RMS
	if (section_index == 0) {
		int lambda_fisher_full_index = lambda_index*JNDIntervalsFull + globaldAIndex;
		if (calculate_ai) FisherAISum[lambda_fisher_full_index] = shared_acc_ai[0] * CUDA_Nerves_Clusters[2*LAMBDA_COUNT+lambda_index];
		if (calculate_rms) F_RA[lambda_fisher_full_index] = shared_acc_rms[0] * CUDA_Nerves_Clusters[2 * LAMBDA_COUNT + lambda_index];
	}
	__syncthreads();

}
// envelope function for GlobalCalculateJND, see detailed description in cochlea_common.h
extern "C" void CudaCalculateJND(
	bool calculate_ai,
	bool calculate_rms,
	int mean_size,
	int fisher_size,
	double SPLRefVal,
	double Fs,
	double scaleBMVelocityForLambdaCalculation,
	double *nIHC,
	int *JND_Calculated_Intervals,
	int numOFJNDCalculated,
	int *JND_Refrence_Intervals,
	int numOFJNDReferences,
	int handeledIntervalsJND,
	int JNDIntervalsFull, // global
	int JNDIntervals, // current input # of handeled intervals
	int JNDIntervalHeadNodes,
	int overlapNodes,
	int JNDIntervalNodes, 
	int lengthOffset, // local not global
	int *JND_Serial_Intervals_Positions,
	int *JND_Interval_To_Reference,
	JNDFloat *F_RA, // result for fisher rate not lambda summed, but time and space reduced
	JNDFloat *FisherAISum, // result for fisher AI not lambda summed, but time and space reduced	
	int writeMatrixSize,
	int calculatedMatrixSize,
	double JND_Delta_Alpha_Length_Factor,
	int JND_USE_Spont_Base_Reference_Lambda,
	int Show_Run_Time,
	bool calcdA,
	bool show_generated_input_params_cuda ,
	int backup_stage,// in case of viewing output id of backup stage
	Log &outer_log
	) noexcept(false) {
	std::cout << "Calculating JND on GPU" << std::endl;
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 16);
	// calculated dA if not already calculated
	if (calcdA) {
		dim3 filtersGriddA(1, 1, 1);
		dim3 filtersThreadsdA(JNDIntervals, 1, 1);
		cudaEventsStartTimer(start, stop, Show_Run_Time & 16);
		cudaCalculateDA KERNEL_ARGS2(filtersGriddA, filtersThreadsdA)(cudaHolderData.cuda_input_samples, cudaJNDHolder.cuda_jnd_params, JNDIntervalNodes, JNDIntervalHeadNodes, lengthOffset, overlapNodes);
		outer_log.timeAtFlag(34, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 16, "dA calculation for JND"), Show_Run_Time & 16);
	}
	// copy paramers to GPU
	cudaEventsStartTimer(start, stop, Show_Run_Time & 16);
	gpuAssert(cudaMemcpy(cudaLambdaHolderData.cuda_nIHC, nIHC, SECTIONS*sizeof(double), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_JND_Refrence_Intervals, JND_Refrence_Intervals, numOFJNDReferences*sizeof(int), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_JND_Serial_Intervals_Positions, JND_Serial_Intervals_Positions, JNDIntervalsFull*sizeof(int), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_JND_Interval_To_Reference, JND_Interval_To_Reference, numOFJNDCalculated*sizeof(int), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_JND_Calculated_Intervals, JND_Calculated_Intervals, numOFJNDCalculated*sizeof(int), cudaMemcpyHostToDevice));
	outer_log.timeAtFlag(35, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 16, "JND Memory preaparations"), Show_Run_Time & 16);
	dim3 filtersGrid(JNDIntervals, LAMBDA_COUNT, 1);
	dim3 filtersThreads(SECTIONS, 1, 1);
	if (show_generated_input_params_cuda) {
		std::cout << "lengthOffset = " << lengthOffset << "\n"
			<< "overlapNodes  = " << overlapNodes << "\n"
			<< "JND_Delta_Alpha_Length_Factor  = " << JND_Delta_Alpha_Length_Factor << "\n"
			<< "calculatedMatrixSize  = " << calculatedMatrixSize << "\n"
			<< "JNDIntervalHeadNodes  = " << JNDIntervalHeadNodes << "\n"
			<< "filtersGrid = " << showDIM3(filtersGrid) << "\n"
			<< "filtersThreads = " << showDIM3(filtersThreads) << "\n";
	}
	if (calculate_rms) {
		// mean rate of lambda (part of Eq. 17) calculate pre run to ensure device synchronization
		cudaEventsStartTimer(start, stop, Show_Run_Time & 16);
		GlobalCalculateMeanRate KERNEL_ARGS2(filtersGrid, filtersThreads)(
			cudaJNDHolder.cuda_jnd_params,
			cudaLambdaHolderData.cuda_JND_Lambda,
			cudaJNDHolder.cuda_MeanRate,
			lengthOffset,
			calculatedMatrixSize,
			JNDIntervals, // local	
			JNDIntervalNodes,
			JNDIntervalHeadNodes,
			overlapNodes,
			JND_USE_Spont_Base_Reference_Lambda
			);
		outer_log.timeAtFlag(36, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 16, "JND Mean Rate array calculation"), Show_Run_Time & 16);
	}
	cudaEventsStartTimer(start, stop, Show_Run_Time & 16);
	// after mean rate ready can calculate the rest
	GlobalCalculateJND KERNEL_ARGS2(filtersGrid, filtersThreads)(
		calculate_ai,
		calculate_rms,
		cudaJNDHolder.cuda_jnd_params,
		cudaLambdaHolderData.cuda_JND_Lambda,
		cudaJNDHolder.cuda_MeanRate,
		cudaLambdaHolderData.cuda_Buffer1,
		cudaLambdaHolderData.cuda_nIHC,
		Fs,
		scaleBMVelocityForLambdaCalculation,
		writeMatrixSize,
		calculatedMatrixSize,
		overlapNodes,
		JNDIntervalHeadNodes,
		JNDIntervalNodes,
		lengthOffset,
		JNDIntervals, // local
		JNDIntervalsFull, // local
		cudaJNDHolder.cuda_JND_Calculated_Intervals,  // global
		numOFJNDCalculated,	// global
		cudaJNDHolder.cuda_JND_Refrence_Intervals, // global
		numOFJNDReferences,	 // global
		handeledIntervalsJND, // already handeled intervals
		cudaJNDHolder.cuda_JND_Serial_Intervals_Positions,
		cudaJNDHolder.cuda_JND_Interval_To_Reference,
		cudaJNDHolder.cuda_F_RA,
		cudaJNDHolder.cuda_FisherAISum,
		JND_Delta_Alpha_Length_Factor,
		cudaJNDHolder.global_device_params,
		calcdA ? 1 : 0,
		JND_USE_Spont_Base_Reference_Lambda, 
		backup_stage
		);
	gpuAssert(cudaMemcpy(F_RA, cudaJNDHolder.cuda_F_RA, fisher_size*sizeof(JNDFloat), cudaMemcpyDeviceToHost));
	gpuAssert(cudaMemcpy(FisherAISum, cudaJNDHolder.cuda_FisherAISum, fisher_size*sizeof(JNDFloat), cudaMemcpyDeviceToHost));
	outer_log.timeAtFlag(37, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 16, "JND calculation"), Show_Run_Time & 16);
}

/**
* generating input per profile
*/
__global__ void CudaGenerateInputFromProfile(
	device_jnd_params *input_profiles,
	float *WN, // white noise array
	float *Signal,
	float *input_samples, // input samples to load
	double wn_dc, // noise dc to be decreased
	double wn_energy_normalize_factor, // factor to normalize energy interval
	int signal_mode,
	double signal_dc, // noise dc to be decreased
	double signal_energy_normalize_factor, // factor to normalize energy interval
	int startProfile,
	int calculatedProfiles,
	int overlapNodes, // for start offset
	int IntervalLength, //number of nodes on actual input
	int JND_Interval_Head,
	float Fs
	) {
	int profile_index = startProfile + blockIdx.y;
	int interval_position = blockIdx.x*blockDim.x + threadIdx.x;
	int input_sample_position = blockIdx.y*IntervalLength + interval_position;
	double dA = input_profiles[profile_index].dA;
	double Wn = input_profiles[profile_index].Wn;
	double frequency = 0;
	if (signal_mode == 0) {
		frequency = input_profiles[profile_index].frequency;
	}
	double time = double(interval_position - JND_Interval_Head) / double(Fs);
	double timeCut = 2 * PI*frequency*time;
	double sum = 0.0;
	if (signal_mode) sum = (dA*(double(Signal[interval_position]) - signal_dc) / signal_energy_normalize_factor);
	else sum = (dA*cos(timeCut) / signal_energy_normalize_factor);
	 sum=sum+ (Wn*(double(WN[interval_position]) - wn_dc) / wn_energy_normalize_factor);
	input_samples[input_sample_position] = float(sum);
}
// calculates Hearing aid effect on the signal, done before BM velocity calculation
__global__ void CudaProcessSignalTroughHearingAID(
	device_jnd_params *input_profiles,
	float *input_samples, // input samples to load
	float *input_samples_auxivulary, // for temporary saving 
	int startProfile,
	int overlapNodes, // for start offset
	int IntervalLength, //number of nodes on actual input
	int JND_Interval_Head,
	float Fs,
	float *fir_transfer_function,
	int fir_transfer_function_length
	) {
	int interval_position = blockIdx.x*blockDim.x + threadIdx.x;
	int input_sample_position = blockIdx.y*IntervalLength + interval_position;
	float function_summary = 0.0f;// input_samples[input_sample_position];
	int backward_positions = min(interval_position, fir_transfer_function_length);
	for (int i = 0; i < backward_positions; i++) {
		function_summary = fmaf(fir_transfer_function[i], input_samples[input_sample_position - i], function_summary);
	}
	input_samples_auxivulary[input_sample_position] = function_summary;
}
// calculates Hearing aid effect on the signal can calculate for IIR filters as well, done before BM velocity calculation
__global__ void CudaProcessSignalTroughHearingAIDIIR(
	device_jnd_params *input_profiles,
	float *input_samples, // input samples to load
	float *input_samples_auxivulary, // for temporary saving 
	int startProfile,
	int overlapNodes, // for start offset
	int IntervalLength, //number of nodes on actual input
	int JND_Interval_Head,
	float Fs,
	float *iir_transfer_function,
	int iir_transfer_function_length
	) {
	int interval_index = blockIdx.x*blockDim.x + threadIdx.x;
	int input_sample_position = interval_index*IntervalLength;
	int input_sample_end_position = input_sample_position + IntervalLength;
	for (int i = input_sample_position; i < input_sample_end_position; i++) {
		int backward_positions = min(i - input_sample_position+1, iir_transfer_function_length);
		float function_summary = input_samples[i];// input_samples[input_sample_position];
		for (int j = 1; j < backward_positions; j++) {
			function_summary = fmaf(iir_transfer_function[j], input_samples[i-j], function_summary);
		}
		input_samples_auxivulary[i] = function_summary;
	}
}

/* Standard C Function: Greatest Common Divisor */
int
gcd(int a, int b) {
	int c;
	while (a != 0) {
		c = a; a = b%a;  b = c;
	}
	return b;
}
double wn_dc;
double wn_energy_normalize_factor;
double signal_dc;
double signal_energy_normalize_factor;
void calculateDCANDNornalizationPostProcess(
	int Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
	double Normalize_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
	int start_dc_expected_value_calculation,
	int end_dc_expected_value_calculation,
	int start_dc_normalized_value_calculation,
	int end_dc_normalized_value_calculation,
	float Fs,
	double &dc,
	double &energy_normalize_factor
) {
	
	if (Normalize_Sigma_Type == 1) {
		// option 1 division factor equal sqrt of the summary, avg energy is normalized
		energy_normalize_factor = energy_normalize_factor / sqrt(static_cast<double>(end_dc_normalized_value_calculation - start_dc_normalized_value_calculation));
	}
	else if (Normalize_Sigma_Type == 2 && Normalize_Energy_To_Given_Interval > 0) {
		// option 2 normalize to given time interval
		energy_normalize_factor = energy_normalize_factor / sqrt(Fs*Normalize_Energy_To_Given_Interval);
	}
	else if (Normalize_Sigma_Type == 3) {
		// option 3 energy not normalized, results identical to option 1 in case of pure tone
		energy_normalize_factor = 1;
	}
}
void calculateDCANDNornalizationFactorPureTone(
	int Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
	double Normalize_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
	double Remove_Generated_DC,//if 1 removes the 0 frequency value from noise
	int start_dc_expected_value_calculation,
	int end_dc_expected_value_calculation,
	int start_dc_normalized_value_calculation,
	int end_dc_normalized_value_calculation,
	float Fs,
	double &dc,
	double &energy_normalize_factor
) {
	dc = 0.0;
	// default, normalizing sigma to 1
	energy_normalize_factor = sqrt(0.5*static_cast<double>(end_dc_normalized_value_calculation - start_dc_normalized_value_calculation));
	calculateDCANDNornalizationPostProcess(
		Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
		Normalize_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
		start_dc_expected_value_calculation,
		end_dc_expected_value_calculation,
		start_dc_normalized_value_calculation,
		end_dc_normalized_value_calculation,
		Fs,
		dc,
		energy_normalize_factor
	);
	//printf("dc=%.4e,energy_normalize_factor=%.4e\n", dc, energy_normalize_factor);
}

void calculateDCANDNornalizationFactor(
	float *Source,
	int Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
	double Normalize_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
	double Remove_Generated_DC,//if 1 removes the 0 frequency value from noise
	int start_dc_expected_value_calculation,
	int end_dc_expected_value_calculation,
	int start_dc_normalized_value_calculation,
	int end_dc_normalized_value_calculation,
	float Fs,
	double &dc,
	double &energy_normalize_factor
	) {
	dc = Source[start_dc_expected_value_calculation];
	for (int idx = start_dc_expected_value_calculation + 1; idx < end_dc_expected_value_calculation; idx++) {
		dc = dc + Source[idx];
	}
	dc = Remove_Generated_DC * dc / static_cast<double>(end_dc_expected_value_calculation - start_dc_expected_value_calculation);
	energy_normalize_factor = Source[start_dc_normalized_value_calculation] * Source[start_dc_normalized_value_calculation];
	for (int idx = start_dc_normalized_value_calculation + 1; idx < end_dc_normalized_value_calculation; idx++) {
		energy_normalize_factor = energy_normalize_factor + ((Source[idx] - dc) * (Source[idx] - dc));
	}
	energy_normalize_factor = sqrt(energy_normalize_factor);
	calculateDCANDNornalizationPostProcess(
		Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
		Normalize_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
		start_dc_expected_value_calculation,
		end_dc_expected_value_calculation,
		start_dc_normalized_value_calculation,
		end_dc_normalized_value_calculation,
		Fs,
		dc,
		energy_normalize_factor
	);
	//printf("dc=%.4e,energy_normalize_factor=%.4e\n", dc, energy_normalize_factor);
}

extern "C" void setupToleranceProfile(
	device_jnd_params *profiles,
	bool is_first_time_for_parameters_set, // for fixing arguments just one time
	float Max_M1_SP_Error_Parameter,
	float Max_Tolerance_Parameter,
	int Relative_Error_Parameters,
	float M1_SP_Fix_Factor,
	float Tolerance_Fix_Factor,
	int Blocks_Per_Interval,
	int from_profile_index,
	int calculatedProfiles
	) noexcept(false)  {
	float host_model_max_m1_sp_tolerance[MAX_NUMBER_OF_BLOCKS];
	float host_max_throw_tolerance[MAX_NUMBER_OF_BLOCKS];
	for (int i = 0; i < calculatedProfiles; i++) {
		int globalProfile = from_profile_index + i;
		float m1_sp_fix_factor = 1.0f;
		float throw_tolerance_factor = 1.0f;
		if (Relative_Error_Parameters > 0) {
			m1_sp_fix_factor = powf(10.0f, M1_SP_Fix_Factor*static_cast<float>(profiles[globalProfile].dBSPLSignal));
			throw_tolerance_factor = powf(10.0f, Tolerance_Fix_Factor*static_cast<float>(profiles[globalProfile].dBSPLSignal));

		}
		for (int j = 0; j < Blocks_Per_Interval; j++) {
			int model_index = Blocks_Per_Interval*i + j;
			host_model_max_m1_sp_tolerance[model_index] = Max_M1_SP_Error_Parameter*m1_sp_fix_factor;
			host_max_throw_tolerance[model_index] = Max_Tolerance_Parameter*throw_tolerance_factor;
		}
	}
	gpuAssert(cudaMemcpy(cudaHolderGeneratedData.generated_model_max_m1_sp_tolerance, host_model_max_m1_sp_tolerance, MAX_NUMBER_OF_BLOCKS*sizeof(float), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemcpy(cudaHolderGeneratedData.generated_model_throw_tolerance, host_max_throw_tolerance, MAX_NUMBER_OF_BLOCKS*sizeof(float), cudaMemcpyHostToDevice));
}
extern "C" void uploadProfiles(
	device_jnd_params *profiles,
	int numOfProfiles,
	bool profilesLoaded // upload profiles array only if false
	) noexcept(false)  {

	if (!profilesLoaded) {
		gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_jnd_params, profiles, numOfProfiles*sizeof(device_jnd_params), cudaMemcpyHostToDevice));
	}
}
extern "C" void generateInputFromProfile(
	device_jnd_params *profiles,
	float *WN, // white noise array, single interval length white noise array, expected power level (linear of 1)
	int wn_length, // max length of white noise    
	float *Signal, //signal array, single interval length white noise array, expected power level (linear of 1)
	int signal_length, // max length of signal noise
	int signal_mode, // 0 - is for normal frequencies, 1 - is for signal array
	int Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
	int Normalize_Sigma_Type_Signal,
	double Normalize_Noise_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
	double Remove_Generated_Noise_DC,//if 1 removes the 0 frequency value from noise
	int start_dc_expected_value_calculation,
	int end_dc_expected_value_calculation,
	int start_dc_normalized_value_calculation,
	int end_dc_normalized_value_calculation,
	int numOfProfiles,
	bool profilesLoaded, // upload profiles array only if false
	int from_profile_index,
	int calculatedProfiles,
	int overlapNodes, // for start offset
	int IntervalLength, //number of nodes on actual input
	int JND_Interval_Head,
	int JND_Interval_Actual_Length,
	float Fs, // sample frequency
	int Show_Generated_Input, // show generated input from file
	float *target_input, // if Show_Generated_Input is true it will copy here the result with nodes fix per position			 
	bool Show_Generated_Configuration, // % for debugging shw profiles of created input	 
	bool is_first_time_for_parameters_set, // for fixing arguments just one time
	float Max_M1_SP_Error_Parameter,
	float Max_Tolerance_Parameter,
	int Relative_Error_Parameters,
	float M1_SP_Fix_Factor,
	float Tolerance_Fix_Factor,
	int Blocks_Per_Interval,
	int Show_Run_Time,
	float *fir_transfer_function,
	int fir_transfer_function_length,
	float *iir_transfer_function,
	int iir_transfer_function_length,
	Log &outer_log
	) noexcept(false) {
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 32);
	cudaEventsStartTimer(start, stop, Show_Run_Time & 32);
	if (!profilesLoaded) {
		gpuAssert(cudaMemcpy(cudaJNDHolder.cuda_jnd_params, profiles, numOfProfiles*sizeof(device_jnd_params), cudaMemcpyHostToDevice));
		gpuAssert(cudaMemcpy(cudaSignalHolder.cuda_WN, WN, wn_length*sizeof(float), cudaMemcpyHostToDevice));
		if (signal_mode) {
			gpuAssert(cudaMemcpy(cudaSignalHolder.cuda_Signal, Signal, signal_length*sizeof(float), cudaMemcpyHostToDevice));
		}
	}
	if (end_dc_normalized_value_calculation == 0) end_dc_normalized_value_calculation = IntervalLength;
	if (end_dc_expected_value_calculation == 0) end_dc_expected_value_calculation = IntervalLength;
	if (is_first_time_for_parameters_set) {
		calculateDCANDNornalizationFactor(
			WN,
			Normalize_Sigma_Type, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
			Normalize_Noise_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
			Remove_Generated_Noise_DC,//if 1 removes the 0 frequency value from noise
			start_dc_expected_value_calculation,
			end_dc_expected_value_calculation,
			start_dc_normalized_value_calculation,
			end_dc_normalized_value_calculation,
			Fs,
			wn_dc,
			wn_energy_normalize_factor
			);
		if (signal_mode) {
			calculateDCANDNornalizationFactor(
				Signal,
				Normalize_Sigma_Type_Signal, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
				Normalize_Noise_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
				Remove_Generated_Noise_DC,//if 1 removes the 0 frequency value from noise
				start_dc_expected_value_calculation,
				end_dc_expected_value_calculation,
				start_dc_normalized_value_calculation,
				end_dc_normalized_value_calculation,
				Fs,
				signal_dc,
				signal_energy_normalize_factor
				);
		}
		else {
			// normalize pure tones to 1 
			calculateDCANDNornalizationFactorPureTone(
				Normalize_Sigma_Type_Signal, // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
				Normalize_Noise_Energy_To_Given_Interval,//  if noise generated normalize energy to given signal
				Remove_Generated_Noise_DC,//if 1 removes the 0 frequency value from noise
				start_dc_expected_value_calculation,
				end_dc_expected_value_calculation,
				start_dc_normalized_value_calculation,
				end_dc_normalized_value_calculation,
				Fs,
				signal_dc,
				signal_energy_normalize_factor
			);
		}
	}
	int threadsPerBlock = IntervalLength / Blocks_Per_Interval;
	int blocksOnxDim = Blocks_Per_Interval;
	if (threadsPerBlock > 1024) {
		int threads_number = static_cast<int>(ceilf(sqrtf(float(threadsPerBlock))));
		while (gcd(threads_number, threadsPerBlock) != threads_number) threads_number++;
		if (threads_number > 1024) {
			threads_number = threadsPerBlock / threads_number;
		}
		blocksOnxDim = Blocks_Per_Interval*threadsPerBlock / threads_number;
		threadsPerBlock = threads_number;
	}

	setupToleranceProfile(
		profiles,
		is_first_time_for_parameters_set, // for fixing arguments just one time
		Max_M1_SP_Error_Parameter,
		Max_Tolerance_Parameter,
		Relative_Error_Parameters,
		M1_SP_Fix_Factor,
		Tolerance_Fix_Factor,
		Blocks_Per_Interval,
		from_profile_index,
		calculatedProfiles
		);

	dim3 filtersGrid(blocksOnxDim, calculatedProfiles, 1);
	dim3 filtersThreads(threadsPerBlock, 1, 1);
	if (Show_Generated_Configuration) {
		std::cout << std::boolalpha << "Show Generated Input: " << Show_Generated_Input << std::endl;
		std::cout << "filtersGrid" << showDIM3(filtersGrid) << std::endl;
		std::cout << "filtersThreads" << showDIM3(filtersThreads) << std::endl;
		std::cout << "IntervalLength = " << IntervalLength << std::endl;
		std::cout << "overlapNodes = " << overlapNodes << std::endl;
		std::cout << "calculatedProfiles = " << calculatedProfiles << std::endl;
		std::cout << "from_profile_index = " << from_profile_index << std::endl;
		std::cout << "Fs = " << Fs << std::endl;
		std::cout << "Normalize_Sigma_Type = " << Normalize_Sigma_Type << std::endl;
		std::cout << "WN(DC) = " << wn_dc << std::endl;
		std::cout << "WN(Normal_Factor) = " << wn_energy_normalize_factor << std::endl;
	}
	
		CudaGenerateInputFromProfile KERNEL_ARGS2(filtersGrid, filtersThreads)(
			cudaJNDHolder.cuda_jnd_params,
			cudaSignalHolder.cuda_WN, // white noise array
			cudaSignalHolder.cuda_Signal,
			cudaHolderData.cuda_input_samples,
			wn_dc,
			wn_energy_normalize_factor, // factor to normalize energy interval
			signal_mode,
			signal_dc,
			signal_energy_normalize_factor,
			from_profile_index,
			calculatedProfiles,
			overlapNodes, // for start offset
			IntervalLength, //number of nodes on actual input
			JND_Interval_Head,
			Fs
			);
		if (fir_transfer_function_length > 1 || (fir_transfer_function_length > 0 && fir_transfer_function[0] != 1) || iir_transfer_function_length > 1 || (iir_transfer_function_length> 0 && iir_transfer_function[0] != 1) ) {
			// processing hear the transfer function
			float *cuda_transfer_function;
			float *cuda_input_samples_auxivulary;
			int processed_input_length = static_cast<int>(filtersGrid.x*filtersGrid.y*filtersGrid.z*filtersThreads.x*filtersThreads.y*filtersThreads.z);
			gpuAssert(cudaMalloc((void **)&cuda_transfer_function, max(fir_transfer_function_length,iir_transfer_function_length)*sizeof(float)));
			gpuAssert(cudaMemcpy(cuda_transfer_function, fir_transfer_function, fir_transfer_function_length*sizeof(float), cudaMemcpyHostToDevice));
			gpuAssert(cudaMalloc((void **)&cuda_input_samples_auxivulary, processed_input_length*sizeof(float)));
			CudaProcessSignalTroughHearingAID KERNEL_ARGS2(filtersGrid, filtersThreads)(
				cudaJNDHolder.cuda_jnd_params,
				cudaHolderData.cuda_input_samples, // input samples to load
				cuda_input_samples_auxivulary, // for temporary saving 
				from_profile_index,
				overlapNodes, // for start offset
				IntervalLength, //number of nodes on actual input
				JND_Interval_Head,
				Fs,
				cuda_transfer_function,
				fir_transfer_function_length
				);
			if (iir_transfer_function_length > 1 || (iir_transfer_function_length > 0 && iir_transfer_function[0] != 1) ) {
				// reverse use of buffer/ input to avoid copying completely
				int threadsIIR = calculatedProfiles;
				int blocksIIR = 1;
				gpuAssert(cudaMemcpy(cuda_transfer_function, iir_transfer_function, iir_transfer_function_length*sizeof(float), cudaMemcpyHostToDevice));
				int threads_number_iir = static_cast<int>(ceilf(sqrtf(float(threadsIIR))));
				while (gcd(threads_number_iir, threadsIIR) != threads_number_iir) threads_number_iir++;
				blocksIIR = threadsIIR / threads_number_iir;
				threadsIIR = threads_number_iir;
				dim3 filtersGridIIR(blocksIIR, 1, 1);
				dim3 filtersThreadsIIR(threadsIIR, 1, 1);
				CudaProcessSignalTroughHearingAIDIIR KERNEL_ARGS2(filtersGridIIR, filtersThreadsIIR)(
					cudaJNDHolder.cuda_jnd_params,
					cuda_input_samples_auxivulary, // input samples to load
					cudaHolderData.cuda_input_samples, // for temporary saving 
					from_profile_index,
					overlapNodes, // for start offset
					IntervalLength, //number of nodes on actual input
					JND_Interval_Head,
					Fs,
					cuda_transfer_function,
					iir_transfer_function_length
					);
			} else {
				gpuAssert(cudaMemcpy(cudaHolderData.cuda_input_samples, cuda_input_samples_auxivulary, processed_input_length*sizeof(float), cudaMemcpyDeviceToDevice));
			}
			
			gpuAssert(cudaFree(cuda_transfer_function));
			gpuAssert(cudaFree(cuda_input_samples_auxivulary));
		}
		if (Show_Generated_Input & 1 > 0) {
			gpuAssert(cudaMemcpy(target_input, cudaHolderData.cuda_input_samples, calculatedProfiles*IntervalLength*sizeof(float), cudaMemcpyDeviceToHost));
		}
	
	
	outer_log.timeAtFlag(43, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 32, "Input Generation"), Show_Run_Time & 32);
}
// fixed lambda values for output
template<typename T> __global__ void CUDAFIXJND_Lambda(
	volatile T *cuda_Lambda,
	volatile T *cuda_Buffer,
	int cuda_buffer_update) {
	int ind = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*blockIdx.y;
	float fix_spikes = CUDA_Nerves_Clusters[blockIdx.y];
	if (cuda_buffer_update) {
		cuda_Buffer[ind] = T(fmaxf(fix_spikes, float(cuda_Lambda[ind])));
	}
	cuda_Lambda[ind] = T(fmaxf(0.0f, float(cuda_Lambda[ind]) - fix_spikes));
	
	__syncthreads();
}


template __global__ void CUDAFIXJND_Lambda<float>(volatile float *cuda_Lambda,
	volatile float *cuda_Buffer,
	int cuda_buffer_update);
template __global__ void CUDAFIXJND_Lambda<double>(volatile double *cuda_Lambda,
	volatile double *cuda_Buffer,
	int cuda_buffer_update);

template<class T> extern void updateCUDALambdaArray(T *lambda_array,T* cuda_buffer, size_t calc_time_nodes, int sections,int Show_Run_Time,int Show_Device_Data,int cuda_buffer_update,Log &outer_log) {
	dim3 grid(calc_time_nodes, LAMBDA_COUNT, 1);
	dim3 thrds(sections, 1, 1);
	if (Show_Device_Data & 16) {
		std::cout << "CUDAFIXJND_Lambda<<<" << showDIM3(grid) << "," << showDIM3(thrds) << " >>>(lambda_array);" << std::endl;
	}
	cudaEvent_t start, stop;
	cudaEventsCreate(start, stop, Show_Run_Time & 32);
	cudaEventsStartTimer(start, stop, Show_Run_Time & 32);
	CUDAFIXJND_Lambda<T> << <grid, thrds >> >(lambda_array, cuda_buffer, cuda_buffer_update);
	outer_log.timeAtFlag(44, cudaEventsStopQueryTimer(start, stop, Show_Run_Time & 32, "Fix Lambda"), Show_Run_Time & 32);
}
