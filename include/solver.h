/******************************************************************\

	File Name :		solver.h
	===========

	Classes Defined :	CVarStep, CSolver
	=================

	Description	:	classes needed for the algorithm solution 
	=============


\******************************************************************/
#pragma once

#ifndef __SOLVER
#define __SOLVER
 
#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include "params.h"
#include "mutual.h"
#include "cvector.h"
#include "bin.h"
#include "TBin.h"
#include "tridiag.h"
#include "model.h"
#include "ode.h"
#include "state.h"
#include "AudioGramCreator.h"
#include "OutputBuffer.h"
#include "memoryTest.h"
#include "Log.h"
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


class CSolver
{

public:

	CParams             *input_params;
	int					num_params;
	double              next_save_time;
	double				_current_time;		// Current state time
	double				_time_step;
	const double		_min_time_step;		// Minimum time step size
	const double		_max_time_step;		// Maximum time step size
	vector<double>		_input_buffer;		// an input buffer for single read of the input
	double				_run_time;			// Total run time.		// QA - need to prtoect it
	long				_total_file_length; // total input file length
	long				_buffer_start_position; // buffer start position of read from file
	CModel*				_model = NULL;				// cochlear model
	CTriDiag			_TriMat;			// tri-diagonal solver
	CState*				_now;				// Set two time states
	CState*				_past;				// Set two time states	 
	cudaDeviceProp deviceProp;
	const int _device_id; //cuda  device id from outside
	//OutputBuffer<float> _output_file; // unified output file
	//const std::string	_generic_extention_filename;	// to ad to all data output files
	//const std::string	_data_files_path;				// path to all data output files
	vector<float>		_WN; // generated white noise array
	vector<float>		_WN_draft; // generated white noise array, copied in order for mathematic manipulations each round
	vector<float>		_Signal; // generated signal for testing
	vector<float>		_Signal_draft; // generated signal for testing, copied in order for mathematic manipulations each round
	std::vector<float> input_samples;
	std::vector<float> res_speeds;
	std::vector<float> convergence_times;
	std::vector<float> convergence_times_blocks;
	std::vector<int> Failed_Converged_Time_Node;
	std::vector<int> Failed_Converged_Blocks;
	std::vector<int> Failed_Converged_Signals;
	std::vector<float> convergence_loops;
	std::vector<float> convergence_loops_blocks;
	std::vector<float> convergence_jacoby_loops_per_iteration;
	std::vector<float> convergence_jacoby_loops_per_iteration_blocks;

	Log solver_log;
	//const std::string	_expected_file_name;	// Input file name.
	//CBin*				_expected_file;

	// Input file:
	TBin<double>*				_input_file;
	
	
	AudioGramCreator*	creator;
	//float *backup_speeds;


	// Functions:
	// ----------
	/*
	CSolver(
		CParams *tparams,
		int num_params,
		const std::string generic_extention_filename, 
		const std::string data_files_path
	);
	*/
	CSolver(CParams *tparams,int num_params, const int device_id);
	CSolver(const int device_id);
	void updateStatus(CParams *tparams, int num_params);
	void init_params(  int );
	void clearModel(void);
	//void release_last_params(  int );
	~CSolver();
	void clearSolver();
	// Solve the model
	void Run(double start_time);
	void Run_Cuda(double start_time);
	void setConstants(void);
	// Clac the tridiagonal and assign the result into _TriMat.
	void Init_Tri_Matrix();		
	// Time step size should be between the allowed values: [MIN_TIME_STEP, MAX_TIME_STEP].
	double Bound_Time_Step( double& time_step );

	// Open all data files for work:
	void Open_Data_Files(int);

	// Save calculated (and desired) data into the harddisk:
	void Save_Results(const long step_counter, const double current_time);
	mapper<std::string, double> compressDeviceData();
	string showDeviceData();
	// Close all open data files (input and output):
	void Close_Data_Files(int);

	// Get the concatenated full file name for a specific diven data:
	void generateLongWN(int params_set_counter); // generating white noise array at the length of longest processed interval
	void preProcessNoise(
		vector<float>& noise_vector,
		int param_set_counter
	);
	inline size_t maxWNLength() { return maxTimeBlocks()*longestTimeBlock(); }
	inline bool continueRunningInputFile(int params_set_counter) { return input_params[params_set_counter].sim_type == SIM_TYPE_VOICE&&(_current_time + input_params[params_set_counter].overlapTime < input_params[params_set_counter].duration); }
	inline bool continueRunningProfileGenerator(int params_set_counter,int from_profile_index) { 
		//if (input_params[params_set_counter].Generating_Input_Profile()) {
		//	std::cout << "from_profile_index(" << from_profile_index << ")<static_cast<int>(input_params[params_set_counter].inputProfile.size())(" << static_cast<int>(input_params[params_set_counter].inputProfile.size()) << ")" << std::endl;
		//}
		return input_params[params_set_counter].Generating_Input_Profile()&&from_profile_index<static_cast<int>(input_params[params_set_counter].inputProfile.size()); 
	}
	/**
	* tests if input buffer is still valid from, 
	*((_current_time+input_params[param_set_counter].intervalTime())*input_params[param_set_counter].Fs) > _buffer_start_position+_input_buffer.size()	&&
	* _run_time > _current_time
	*/
	bool reloadBufferFromFile(int params_set_counter);


	/**
	*	to ensure samples buffer long enough to use on all diffrent configurations
	*	max time blocks will give the largest number of time blocks
	*/
	size_t maxTimeBlocks();


	/**
	*	to ensure samples buffer long enough to use on all diffrent configurations
	*	max time blocks will give the largest time block (with the extra space for last block saved)
	*/
	size_t longestTimeBlock();


	/**
	*	inputBufferTimeNodes will give 	longestTimeBlock()*(maxTimeBlocks()+1), enough length for input samples	at time blocks +1
	*/
	size_t inputBufferTimeNodes();

	/**
	*	bufferResultsSpeeds will give bufferTimeNodes()*SECTIONS for large enough data to solve results speeds on all configurations
	*/
	size_t bufferResultsSpeeds();

	/**
	*	lambdaBufferSpeeds will give bufferResultsSpeeds()*LAMBDA_COUNT, to save all lambdas data
	*/
	size_t bufferLambdaSpeeds();

	/**
	*	bufferLastDataSaved will give MAX(params.totalResultNodesExtended()-params.totalResultNodes())*SECTIONS
	*	a buffer large enough for all configurations
	*/
	//int bufferLastDataSaved();


	/**
	 * calculates maximum overlap nodes in order to add it to time nodes result allocation
	 */
	size_t bufferOverlapNodes();

	inline size_t getCalcTime(size_t write_time) {
		size_t calcTime = write_time;
		if (calcTime%THREADS_PER_IHC_FILTER_SECTION > 0) {
			calcTime += THREADS_PER_IHC_FILTER_SECTION - calcTime%THREADS_PER_IHC_FILTER_SECTION;
		}
		return calcTime;
	}
	// get output write time for partial output
	inline size_t getWriteTime(const CParams& params, double remain_time) {
		return min(params.totalTimeNodes(), params.calcOverlapNodes() + static_cast<size_t>(ceil(remain_time*params.Fs)));
	}
	// get output time for interval
	inline size_t getWriteTime(const CParams& params) {
		return getWriteTime(params, params.intervalTime());
	}
};



#endif