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



#ifndef COCHLEA_COMMON_H
#define COCHLEA_COMMON_H
#include <driver_types.h>
# include "const.h"	
#include "Log.h"
typedef unsigned int uint;

#ifdef __CUDACC__
    typedef float2 fComplex;
#else
    typedef struct{
        float x;
        float y;
    } fComplex;
#endif

// JND parametrs on GPU
typedef struct device_params_structs {
	int filter_b_start_index; // gives the index on device_time_filter that contains filter b size
	int filter_a_start_index; // gives the index on device_time_filter that contains filter a size
	int cochlea_sections; // param 0 is number of indexes in space
	int calcTime; // param 1 is calculation size in time
	int writeTime; // param 2 is write size in time
	int time_blocks; // param 3 is number of seperate threads calculate each single space index
	int order_of_dc_filter; // param 4 is order of dc filter
	int order_of_ac_filter; // param 5 is order of ac filter
	int lambda_count; // param 6 is order of ac filter
	int time_block; // param7=param1/param3 is time block size
	int lambda_offset; // param 8 is lambda offset fix
	int ovelapNodes; // for testing if filter can look behind its input interval
	int FilterDecoupledMode; // if true than output blocks will not use input with time start befire output block start
	double Lambda_SAT;
	int intervalTimeNodes; // if decoupled this is the size of interval node
	float reverseSQRTScaleBMVelocityForLambdaCalculation;
	int Decouple_Filter;
} device_params;
typedef struct vectors_sum_linear_coefficents_type {
	double A_coefficent;
	double B_coefficent;
	int reverseCoefficents;
} vectors_sum_linear_coefficents;

typedef struct device_jnd_params_structs {
	double dA;  // power linear value
	double recipdA; // reciprocal dA   
	double dBSPLSignal;// power dB value
	double dBSPLNoise;// power dB value
	double Wn; // white noise power
	bool isReference;   // true if its reference interval
	int calculateIndexPosition;
	int referenceIndex;
	double frequency; // input frequency in HZ
} device_jnd_params;
//lambdaFloat *cuda_DC;
//lambdaFloat *cuda_IHC;
//lambdaFloat *cuda_PRE_IHC;
//lambdaFloat *cuda_AC;
//lambdaFloat *cuda_dS;
//lambdaFloat *cuda_Shigh;
//float *BM_internal;
//extern "C" device_jnd_params *cuda_jnd_params;
//JNDFloat *cuda_dMeanRate;
//JNDFloat *cuda_AvgMeanRate;
//JNDFloat *cuda_preFisherAI;
//JNDFloat *cuda_CRLB_RA;
//JNDFloat *cuda_preFisherAITimeReduced;
//extern "C" JNDFloat *cuda_JND_Lambda;
//extern "C" JNDFloat *cuda_Buffer1;
//extern struct cudaLambdaHolderData;
//JNDFloat *cuda_dLambda;

#ifdef __cplusplus
extern "C" {
#endif

/**
* functions use for timing analysis
*
*
*/


void cudaEventsCreate(cudaEvent_t& start, cudaEvent_t& stop, int condition) throw(...);
void cudaEventsStartTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition) throw(...);
void cudaEventsStopTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition) throw(...);
float cudaEventsStopQueryTimer(cudaEvent_t& start, cudaEvent_t& stop, int condition, const std::string& prefix) throw(...);
JNDFloat *getCudaBuffer();
JNDFloat *getCudaLambda();
void loadAihc(float *Aihc) throw(...);
void enableloadAihc() throw(...);
void allocateBuffer(const int size_in_nodes, int Show_Run_Time, cudaEvent_t& start,
	cudaEvent_t& stop,
	cudaDeviceProp deviceProp) throw(...);
void releaseBuffer(int Show_Run_Time, cudaEvent_t& start,
	cudaEvent_t& stop,
	cudaDeviceProp deviceProp) throw(...);
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
 
/**
* Run calculation of BM velocity on the GPU
* input_sample - input signal to process
* time_step/time_step_out - time intervals between samples
* delta_x - cochlea section length
* enable_psi - outdated
* base_index - start input read, should be always 0
* Ts - identical time_step
* _ohc_alpha_l,_ohc_alpha_s,model_Gme, model_a0,model_a1,model_a2,sigma_ow,eta_1,eta_2 - physical parameters from Professor furst article Cochlear Model for Hearing Loss, Table 1, additional data can be viewed on CModel class
* 
*/
void BMOHCNewKernel(

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
int Time_Blocks, // cuda blocks to execute
int samplesBufferLengthP1, // input length include padding
int overlap_nodes_for_block, // input nodes calculated by 2 consecutive cuda blocks for transient effects dissipation
long overlapTimeMicroSec, // ovelap time for blocks in microseconds
int show_transient, // output transient time
float cuda_max_time_step, // max time step allowed for cuda (should be no more than 1e-6 seconds)
float cuda_min_time_step, // minimum time step allowed for cuda (in advised to be less than 1e-14 seconds, despite theoretical 1e-15 seconds effectiveness)
int Decouple_Filter, // number of cuda blocks for single signal 
float Max_M1_SP_Error_Parameter, // tolerance parameter for convverging
float Max_Tolerance_Parameter, // tolerance parameter for convverging
int Relative_Error_Parameters, // tolerance parameter for convverging
int calculate_boundary_conditions, // if true will calculate max tolerance and max m1 sp error from input, should be used if input is not generated within the program	
float M1_SP_Fix_Factor, // tolerance parameter for convverging
float Tolerance_Fix_Factor, // tolerance parameter for convverging
float SPLREfVal, // power for 0 dB signal in dyn units
int Show_Calculated_Power, // debug data control, flag array
int Show_Device_Data, // debug data control, flag array
int Show_Run_Time, // debug data control, flag array
int JACOBBY_Loops_Fast, // number of jcoby loops to perform on fast approximation
int JACOBBY_Loops_Slow, // number of jcoby loops to perform on slow approximation  
int Cuda_Outern_Loops, // max control loops
int Run_Fast_BM_Calculation, // will run BM calculation with relaxed memory requirements
int BMOHC_Kernel_Configuration, // control inner algorithm for BM calculation, use 17 is advised
cudaEvent_t& start, // timer start
cudaEvent_t& stop, // timer stop
cudaDeviceProp deviceProp, // take device properties
Log &outer_log // write to outer log
) throw(...);
 

void BMOHCKernel_Wait_Threads() throw(...);
void BMOHCKernel_Copy_Results(float *target, size_t resultNodes, size_t offset) throw(...);
extern void extractConvergenceTimes(float *convergence_times, size_t nodes);

#ifdef __cplusplus
}
#endif
template<class T> extern void GeneralKernel_Copy_Results_Template(T *target, T *src, size_t size);
template<class T> extern void GeneralKernel_Copy_Results_Template(T *target, T *src, size_t size, size_t offset);
template<class T> extern void ReverseKernel_Copy_Results_Template(T *cpu_src, T *cuda_target, size_t start_time_node, size_t time_nodes, int sections);


template<class T> extern void updateCUDALambdaArray(T *lambda_array, T *cuda_buffer, size_t calc_time_nodes, int sections, int Show_Run_Time, int Show_Device_Data, int cuda_buffer_update,Log &outer_log);

#ifdef __cplusplus
extern "C" {
#endif
/**
* src is __global__ array in cuda device
* target is array in host
* size is array length
* function will copy from src to taraget use cudaMemcpy
*/
JNDFloat *getCudaMeanRate();
// function to copy results on GPU to CPU target arrays
void GeneralKernel_Copy_Results(float *target, float *src, size_t size) throw(...);
void GeneralKernel_Copy_Results_Double(double *target, double *src, size_t size) throw(...);
void ReverseKernel_Copy_Results(float *src, size_t size) throw(...);
void BMOHCKernel_Copy_Lambda(JNDFloat *target, size_t lambdaNodes, int offset) throw(...);
// set values for physical parmeters for the cochlea, view model.h file for exact physical values description
void BMOHCKernel_Init(
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
size_t inputBufferNodes, // input length
size_t resultBufferNodes, // BM velocity length
size_t lambdaBufferNodes, // lambda length
bool first_time, // is this the first time running on this input
int Show_Run_Time, // debug control
int Show_Device_Data, // debug control
Log &outer_log // target for logs
								 ) throw(...);


/** 
	allocate and copy parameters to GPU for IHC calculations 
	AC filter will be multiplied by CONVERT_CMPerSeconds_To_MetersPerSecond
	in order to save me from wasting operations on it.
*/
void IHCNewKernel(
	double *IHC_Damage_Factor,
	float Nerves_Clusters[3*LAMBDA_COUNT], // Aihc parametrs
	double *dc_filter, // filter for DC stage
	int order_of_dc_filter, // length of DC stage
	double *ac_b_filter, // AC filter b array
	double *ac_a_filter, // AC filter a array (if IIR)
	bool is_ac_iir_filter, // does AC stage filter is IIR?
	int order_of_ac_filter, // length of AC filter
	int cochlea_sections, // number of cochlear sections (256 in this program)
	int time_blocks, // number of cuda blocks to execute
	double SPLRefVal, // power for 0 dB signal in dyn units
	float *backup_speeds, // output of backup data, its outdated
	int backup_speeds_length, // outdated
	int calcTime, // total node to calculate
	int writeTime, // total node to output
	int allocateTime, // total nodes to allocate
	int intervalTimeNodes, // single time block time nodes
	int max_backup_nodes_len, // outdated
	int lambda_offset, // offset of time nodes in order to compensate for larger lambda than necessary 
	float Lambda_SAT, // AN saturation level
	float eta_AC, // IHC AC coupling [V/s/cm]
	float eta_DC, // IHC DC coupling [V/s/cm]
	bool first_time, // first running allocate arrays
	bool first_time_for_param_set, // first time on this configuration to reset some parametrs
	bool loadedFromHD, // if data loaded from HD, outdated
	bool disable_advanced_memory_handling, // outdated
	bool review_memory_handling, // debug control
	bool asMemoryHandlingOnly, // debug control
	float scaleBMVelocityForLambdaCalculation,// params[params_set_counter].scaleBMVelocityForLambdaCalculation
	bool CalculateJNDOnGPU, // should be true if JND calculated at all
	int maxJNDIntervals, // JND intervals to calculate at most
	int overlapNodes, // overlap nodes to ignore due to transient effects
	int Decouple_Filter, // filter is decoupled if this parameter largeer than 0,if filter decoupled than output blocks will not use input with time start before output block start
	int Show_Run_Time, // debug control
	Log &outer_log
	) throw(...);
// thi is the calculating IHC functions, its execute multiple kernels, description in cochlea.cu
void RunIHCKernel(JNDFloat *host_backup,int Show_Run_Time,int save_lambda,int backup_stage, int Decouple_Unified_IHC_Factor,Log &outer_log) throw(...);
void cudaMallocatingByMode(void **ptr,size_t bytes_num,bool disable_advanced_mode) throw(...);
// memory release
void IHCKernel_Free() throw(...);
void BMOHCKernel_Free() throw(...);
// Input generation on GPU profile, this creates multiple pitches(or single loaded signal) at multiple powers with multiple noise levels
void InputProfilesArrayInitialization(
	int maxJNDIntervals, // max number of JND intervals
	int wn_length, // white noise nodes
	int signal_length, // signal nodes - outdated
	int signal_mode,
	int Show_Generated_Input) throw(...);
// log GPU status
void viewGPUStatus(int flags, const std::string& prefix) throw(...);
void InputProfilesArrayTermination() throw(...);
// Calculate JND for all information and RMS methods
void CudaCalculateJND(
	bool calculate_ai, //calculate All Information
	bool calculate_rms, // Calculate RMS
	int mean_size, // number of averaged nodes
	int fisher_size, // number of fisher nodes
	double SPLRefVal, // power for 0 dB signal in dyn units
	double Fs, // Sampling Frequency
	double scaleBMVelocityForLambdaCalculation, // factor correction for AN response fix BM velocity from cm/s to m/s
	double *nIHC, // IHC damage per section
	int *JND_Calculated_Intervals, // indices of JND calculated intervals (not pure noise)
	int numOFJNDCalculated, // number of calculated JND intervals
	int *JND_Refrence_Intervals, // Indices of Intervals of pure noise as reference
	int numOFJNDReferences, // number of JND references
	int handeledIntervalsJND,
	int JNDIntervalsFull, // global	 
	int JNDIntervals, // current input # of handeled intervals
	int JNDIntervalHeadNodes,
	//JNDFloat *CRLB_RA,
	//JNDFloat *preFisherAITimeReduced,
	int overlapNodes,
	int JNDIntervalNodes,
	int lengthOffset, // local not global
	int *JND_Serial_Intervals_Positions,
	int *JND_Interval_To_Reference,
	JNDFloat *F_RA, // result for fisher rate not lambda summed, but time and space reduced
	JNDFloat *FisherAISum, // result for fisher AI not lambda summed, but time and space reduced
	//JNDFloat *AvgMeanRate, // averaged mean rate across space to find close to zero points
	int writeMatrixSize, // output matrix size
	int calculatedMatrixSize, // calculated matrix size, includes transient effects
	double JND_Delta_Alpha_Length_Factor, // fix factor to calculate JND as if the input has certain length in seconds, relevant to Eq. 17 in Cochlear Model for Hearing Loss
	int JND_USE_Spont_Base_Reference_Lambda, // outdated
	int Show_Run_Time, // debug control
	bool calcdA, // calc signal power
	bool show_generated_input_params_cuda, // debug
	int backup_stage, // debug
	Log &outer_log
	) throw(...);

/* disabled mathematically incorrect
extern "C" void BMOHCKernel_Init_consts(
	float *mass,
	float *rmass,
	float *R,
	float *S,
	float *Q,
	float *S_ohc,
	float *S_tm,
	float *R_tm,
	const cudaDeviceProp& deviceProp
	);
	*/
// setting individual convergence parmeters for blocks, experimental
void setupToleranceProfile(
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
	) throw(...);
// upload JND profiles to GPU
void uploadProfiles(
	device_jnd_params *profiles,
	int numOfProfiles,
	bool profilesLoaded // upload profiles array only if false
	) throw(...);

int *getCudaFailedTimeNodes();
int *getCudaFailedBlocks();
float *getCudaConvergeSpeedBlocks();
float *getCudaConvergedTimeNodes();
float *getCudaConvergedBlocks();
float *getCudaConvergedJacobyLoopsPerIteration();
float *getCudaConvergedJacobyLoopsPerIterationBlocks();
/**
* generating Input signal on GPU for all powers of signals and noises
*/
void generateInputFromProfile(
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
	int numOfProfiles, // number of profiles for total claculations
	bool profilesLoaded, // upload profiles array only if false
	int from_profile_index, // profile to start calculation from
	int calculatedProfiles,	// calculated profiles in single cuda run
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
	float *fir_transfer_function,	// the b coefficents of the tf
	int fir_transfer_function_length,
	float *iir_transfer_function,	// the a coefficents of the tf
	int iir_transfer_function_length,
	Log &outer_log
	) throw(...);


#ifdef __cplusplus
}
#endif


/*
extern "C" void JaccobyKernel(
float *P,
float *M,
float *U,
float *L,
float *V
							   );  */
 

#endif //COCHLEA_COMMON_H
