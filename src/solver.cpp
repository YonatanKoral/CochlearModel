
#include "stdafx.h"
#include "solver.h"
#include "AudioGramCreator.h"
// c'tor
 
# define INIT_CURR_TIME INIT_TIME_STEP 
CSolver::~CSolver()
{
	clearModel();
}

void CSolver::clearModel(void) {
	if (_model != NULL) {
		delete _model;
		_model = NULL;
	}
}
void  CSolver::setConstants(void) {
	_current_time =  INIT_CURR_TIME;
	_time_step =  INIT_TIME_STEP;
}

CSolver::CSolver(const int device_id) :
	_min_time_step(MIN_TIME_STEP),
	_max_time_step(MAX_TIME_STEP),
	_TriMat(SECTIONS),
	_device_id(device_id) {
	_input_buffer = vector<double>(INPUT_MAX_SIZE);
	setConstants();
	cudaGetDeviceProperties(&deviceProp, _device_id);
	creator = new AudioGramCreator();

}

CSolver::CSolver(CParams *tparams, int num_params, const int device_id) : CSolver(device_id)
	//_expected_file_name( "Data\\expected.bin"),		
	//_expected_file_name( exp_filename) 
{
	//cout << "hasGeneral debug expect #" << num_params << "sets\n";
	
	updateStatus(tparams, num_params);
}

void CSolver::updateStatus(CParams *tparams, int num_params) {
	this->num_params = num_params;
	next_save_time = 0.0;
	input_params = tparams;
	//int allocateTime = inputBufferTimeNodes();
	//int writeTime = getWriteTime(input_params[0]);

	_model = new CModel(tparams, num_params);
	//auto input_samples = std::vector<float>();
	//creator = new AudioGramCreator(getCalcTime(writeTime), writeTime, allocateTime, SECTIONS, input_params, _model, 0.0, input_samples);// bufferLastDataSaved());
}


mapper<std::string, double> CSolver::compressDeviceData() {
	mapper<std::string, double> device_data = mapper<std::string, double>();
	device_data.add("name",deviceProp.name);
	device_data.add("major", deviceProp.major);
	device_data.add("minor", deviceProp.minor);
	device_data.add("Architecture", concat(deviceProp.major,".",deviceProp.minor));
	device_data.add("ConstantMemoryKB", static_cast<double>(deviceProp.totalConstMem / 1024));
	device_data.add("GlobalMemoryMB", static_cast<double>(deviceProp.totalGlobalMem / 1048576));
	device_data.add("sharedMemPerBlockKB", static_cast<double>(deviceProp.sharedMemPerBlock / 1024));
	device_data.add("sharedMemPerMultiprocessorKB", static_cast<double>(deviceProp.sharedMemPerMultiprocessor / 1024));
	device_data.add("warpSize", static_cast<double>(deviceProp.warpSize));
	device_data.add("memoryBusWidth", static_cast<double>(deviceProp.memoryBusWidth));
	device_data.add("localL1CacheSupported", static_cast<double>(deviceProp.localL1CacheSupported));
	device_data.add("globalL1CacheSupported", static_cast<double>(deviceProp.globalL1CacheSupported));
	device_data.add("maxThreadsPerBlock", static_cast<double>(deviceProp.maxThreadsPerBlock));
	device_data.add("maxThreadsDimX", static_cast<double>(deviceProp.maxThreadsDim[0]));
	device_data.add("maxThreadsDimY", static_cast<double>(deviceProp.maxThreadsDim[1]));
	device_data.add("maxThreadsDimZ", static_cast<double>(deviceProp.maxThreadsDim[2]));
	device_data.add("maxThreadsPerMultiProcessor", static_cast<double>(deviceProp.maxThreadsPerMultiProcessor));
	device_data.add("regsPerMultiprocessor", static_cast<double>(deviceProp.regsPerMultiprocessor));
	device_data.add("multiProcessorCount", static_cast<double>(deviceProp.multiProcessorCount));
	device_data.add("clockRateMHZ", static_cast<double>(deviceProp.clockRate / 1024));
	device_data.add("memoryRateMBitsps", static_cast<double>(deviceProp.memoryBusWidth*deviceProp.memoryClockRate / 1024));
	return device_data;
}
std::string CSolver::showDeviceData() {
	ostringstream oss("");
	oss.setf(oss.boolalpha);
	oss << "Device " << _device_id << ": " << deviceProp.name << endl;
	oss << "Architecture Version: " << deviceProp.major<<"."<<deviceProp.minor << endl;
	oss << "Clock frequency: " << (deviceProp.clockRate / 1024) << " MHz"<<endl;
	oss << "warp size: " << deviceProp.warpSize << endl;
	oss << "global memory available: " << (deviceProp.totalGlobalMem / 1048576) << " MBytes" << endl;
	oss << "constant memory available: " << (deviceProp.totalConstMem / 1024) << " KBytes" << endl;
	oss << "shared memory per block available: " << (deviceProp.sharedMemPerBlock / 1024) << " KBytes" << endl;
	oss << "shared memory per multiprocessor available: " << (deviceProp.sharedMemPerMultiprocessor / 1024) << " KBytes" << endl;
	oss << "L2 Cahce size: " << (deviceProp.l2CacheSize / 1024) << " KBytes" << endl;
	oss << "Memory bus clock: " << (deviceProp.memoryClockRate / 1024 ) << " MHz" << endl;
	oss << "Memory bus width: " << (deviceProp.memoryBusWidth) << " Bits" << endl;
	oss << "Memory bus rate: " << (deviceProp.memoryBusWidth*deviceProp.memoryClockRate / 1024) << " MBits/s" << endl;
	oss << "Number of multiprocessors: " << deviceProp.multiProcessorCount  << endl;
	oss << "Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
	oss << "Max threads dimension: " << deviceProp.maxThreadsDim << endl;
	oss << "Max threads per multi processor: " << deviceProp.maxThreadsPerMultiProcessor << endl;
	oss << "Max registers per block: " << deviceProp.regsPerBlock << endl;
	oss << "Max registers per multi processor: " << deviceProp.regsPerMultiprocessor << endl;
	oss << "Number of multiprocessors: " << deviceProp.multiProcessorCount << endl;
	oss << "Does global L1 Cahce supported? " << (deviceProp.globalL1CacheSupported ==1) << endl;
	oss << "Does local L1 Cahce supported? " << (deviceProp.localL1CacheSupported == 1) << endl;
	//oss << "Device warp size: " << deviceProp.warpSize << endl;
	//oss << "Device frequency: " << (deviceProp.clockRate / 1024) << " MHz";
	return oss.str();
}
/**
void CSolver::release_last_params(int  params_set_counter) {
	Close_Data_Files(params_set_counter);
	delete _now;
	delete _past;
}
*/
void CSolver::init_params( int params_set_counter ) 	 	 
{
	_buffer_start_position = 0;
	setConstants();
	// Open all data files for work:
	Open_Data_Files(params_set_counter);
	
	if (input_params[params_set_counter].sim_type == SIM_TYPE_VOICE) {
		if (input_params[params_set_counter].Has_Input_Signal()) {
			_run_time = static_cast<double>(input_params[params_set_counter].Input_Signal.size()) / input_params[params_set_counter].Fs;	// [sec]
		} else {
			_run_time = input_params[params_set_counter].numberOfJNDIntervals()*input_params[params_set_counter].JND_Interval_Duration;
		}
	}
	else
	{
		_run_time = REF_RUN_TIME; //_input_file->_file_length / _Fs;	// [sec]	
	}

	// Prepare the tridiagonal matrix:
	Init_Tri_Matrix();	
	// Creat new states:
	//9a) _now  = new CState( _time_step,		_time_step, _Fs, _TriMat, _input_file, _model );
	//9a)  _past = new CState( _current_time,	_time_step, _Fs, _TriMat, _input_file, _model );
	if ( params_set_counter > 0 ) {
		delete _now;
		delete _past;
	}
	Close_Data_Files(params_set_counter);
	_now = new CState(_time_step, _time_step, input_params[params_set_counter].Fs, _TriMat, &_input_buffer,_buffer_start_position/*, _expected_file*/, *_model, params_set_counter);
	_past = new CState(_current_time, _time_step, input_params[params_set_counter].Fs, _TriMat, &_input_buffer, _buffer_start_position/*, _expected_file*/, *_model, params_set_counter);
	
}



// Clac the tridiagonal and assign the result into _TriMat.
void CSolver::Init_Tri_Matrix()	
{

	double dx_pow2 = _model->_dx * _model->_dx;

	vector<double> UpperDiag(_model->_sections, 1);
	vector<double> MidDiag = -1.0 * (2.0 + dx_pow2 * _model->_Q);
	vector<double> LowerDiag(_model->_sections, 1);
	
	// Middle Diagonal Boundries:
	MidDiag[0]	= -1.0 * ( 1.0 + _model->_dx * _model->_a0 );						
	MidDiag[_model->_sections-1] = 1.0;

	// Lower Diagonal Boundries:
	LowerDiag[_model->_sections-1] = 0.0;		
	LowerDiag[_model->_sections-2] = 0.0;		// as in CM_Miriam_cpp_QA, solver.m, Line 147.

	_TriMat.SetLowerDiag( LowerDiag );
	_TriMat.SetMidDiag( MidDiag );
	_TriMat.SetUpperDiag( UpperDiag );

}


// Time step size should be between the allowed values: [MIN_TIME_STEP, MAX_TIME_STEP].
double CSolver::Bound_Time_Step( double& time_step )
{
	if (time_step < _min_time_step) {
		//time_step = _min_time_step;
		
		cout<<"CSolver::Bound_Time_Step( double& time_step ) - Stability Error - Too small time step"<<endl;
		throw std::runtime_error("CSolver::Bound_Time_Step( double& time_step ) - Stability Error - Too small time step - 20");
		//MyError stability_err("CSolver::Bound_Time_Step( double& time_step ) - Stability Error - Too small time step", "CSolver");
		//throw stability_err;
	}
	else if (time_step > _max_time_step) {
		time_step = _max_time_step;
	}

	return time_step;

}


void CSolver::Run(double start_time)
{
	
	this->Run_Cuda(start_time);

}


void CSolver::generateLongWN(int params_set_counter) {
	int wn_length = maxWNLength();
	_WN = vector<float>(wn_length);
	_WN_draft = vector<float>(wn_length);
	random_device rnd_device;
	default_random_engine  mersenne_engine(rnd_device());
	normal_distribution<float> dist(0, 1); // expected value is 1
	auto gen = std::bind(dist, mersenne_engine);
	generate(begin(_WN), end(_WN), gen);
	copy(begin(_WN), end(_WN), begin(_WN_draft));
	if (input_params[params_set_counter].Show_Generated_Input & 4) {
		cout << "E(Wn) = " << (sum<float>(_WN) / _WN.size()) << "\n";
		cout << "Sigma(Wn) = " << (sum<float>(_WN*_WN) / _WN.size()) << "\n";
	}
}

void CSolver::preProcessNoise(
	vector<float>& noise_vector,
	int param_set_counter
	) {
	vector<float> temp_vector(noise_vector.size(), 0.0);
	// _model->configurations[param_set_counter]._noise_filter
	IIRFilterTemplate<float, float, double>(noise_vector, temp_vector, _model->configurations[param_set_counter]._noise_filter.Denominator, _model->configurations[param_set_counter]._noise_filter.Numerator, 1,0, static_cast<int>(noise_vector.size()), 1);
	//copy(begin(temp_vector), end(temp_vector), begin(noise_vector));
	noise_vector = temp_vector;
	
}
void CSolver::Run_Cuda(double start_time)
{

	
	//printf("running cuda...\n");
	vector<double>	_now_BM_speed_ref(SECTIONS, 0);				// reference speed vector for testing.
	vector<double>	_now_BM_acceleration_ref(SECTIONS, 0);		// reference acceleration vector for testing.
	cudaEvent_t start, stop;
	bool is_first = true;
	double remain_time = 0; // will use to calculate remain time in order to write less data if able
	bool is_first_for_param_set = true;
	int first_max_iter = 1;
	double out_time = 0.0;
	int from_profile_index = 0;
	double start_time_backup = start_time;
	viewGPUStatus(input_params[0].Show_Device_Data,"Run Cuda Start");
	// initialize input and output arrays for cpu side, resize I/O arrays if necessary
	if (input_samples.empty()) {
		input_samples = std::vector<float>(inputBufferTimeNodes(), 0.0f);
		Failed_Converged_Time_Node = std::vector<int>(inputBufferTimeNodes(), 0);
		convergence_times = std::vector<float>(inputBufferTimeNodes(), 0);
		convergence_loops = std::vector<float>(inputBufferTimeNodes(), 0);
		convergence_jacoby_loops_per_iteration = std::vector<float>(inputBufferTimeNodes(), 0);
	}
	else if (input_samples.size() < inputBufferTimeNodes() ) {
		input_samples.resize(inputBufferTimeNodes(), 0.0f);
		Failed_Converged_Time_Node.resize(inputBufferTimeNodes(), 0);
		convergence_times.resize(inputBufferTimeNodes(), 0.0f);
		convergence_loops.resize(inputBufferTimeNodes(), 0.0f);
		convergence_jacoby_loops_per_iteration.resize(inputBufferTimeNodes(), 0.0f);
	}
	if (Failed_Converged_Blocks.empty()) {
		Failed_Converged_Blocks = std::vector<int>(maxTimeBlocks(), 0);
		convergence_times_blocks = std::vector<float>(maxTimeBlocks(), 0);
		convergence_loops_blocks = std::vector<float>(maxTimeBlocks(), 0);
		convergence_jacoby_loops_per_iteration_blocks = std::vector<float>(maxTimeBlocks(), 0);
	}
	else if (Failed_Converged_Blocks.size() < maxTimeBlocks()) {
		Failed_Converged_Blocks.resize(maxTimeBlocks(), 0);
		convergence_times_blocks.resize(maxTimeBlocks(), 0.0f);
		convergence_loops_blocks.resize(maxTimeBlocks(), 0.0f);
		convergence_jacoby_loops_per_iteration_blocks.resize(maxTimeBlocks(), 0.0f);
	}
	// cochlea paremetrs per section
		float Rd[SECTIONS];
		float Sd[SECTIONS];
		float Qd[SECTIONS];
 
		float S_ohcd[SECTIONS];
		float S_tmd[SECTIONS];
		float R_tmd[SECTIONS];
		float gammad[SECTIONS];

		float mass[SECTIONS];

		float rsmm[SECTIONS]; // re3siprocal of mass to replace division with multiplication on gpu
		float uu[SECTIONS];
		float ll[SECTIONS];
		// define BM velocity array, rersize if necessary
		if (res_speeds.empty()) {
			res_speeds = std::vector<float>(bufferResultsSpeeds());
		}
		else if (res_speeds.size() < bufferResultsSpeeds()) {
			res_speeds.resize(bufferResultsSpeeds(), 0.0f);
		}
		

		FILE *amp_file;
		bool generatedInput = _model->firstGeneratedInputSet() != PARAM_SET_NOT_EXIST;
		bool initInputsPreArrays = _model->firstCartesicMultiplicationOfInputsSet() != PARAM_SET_NOT_EXIST || _model->firstGeneratedInputSet() != PARAM_SET_NOT_EXIST || _model->firstJNDCalculationSetONGPU() != PARAM_SET_NOT_EXIST;
		// generate noise arrays, generate white noise values if ineternal white noise selected
		if ( initInputsPreArrays ) {
			InputProfilesArrayInitialization(_model->maxJNDIntervals(), generatedInput ? maxWNLength() : 0, generatedInput ? maxWNLength() : 0, generatedInput, input_params[0].Show_Generated_Input);
		}
		if ( generatedInput ) {
			generateLongWN(0);
		}
	// executed profiles dtetermined before, supports multiple profiles (although Matlab will execute single profile)
	for ( int params_set_counter=0;params_set_counter<_model->_params_counter;params_set_counter++) {
		solver_log.resetLogRound();
		// clears aihc to reload at start of parameter set
		enableloadAihc();
		//_output_file = OutputBuffer<float>(input_params[params_set_counter].output_file_name, __tmin(static_cast<int>(input_params[params_set_counter].Fs*(input_params[params_set_counter].duration + 1.0f)*SECTIONS*(input_params[params_set_counter].disable_lambda?1:4)), MAX_OUTPUT_BUFFER_LENGTH), false);
		
		/**
		* since multiple params set are running on the same solver on the second and upward param set
		* some vectors will need to be reinitialized
		*/
		if ( params_set_counter > 0 ) {
			init_params(params_set_counter);
			start_time = start_time_backup;
			from_profile_index = 0;
			out_time = 0.0;
			first_max_iter = 1;
			is_first_for_param_set = true;
			//fclose(amp_file);
			//sprintf_s<MAX_BUF_LENGTH>(amp_file_name, "Data\\Amplitudes%d.txt", params_set_counter);
		} 
		amp_file = NULL;
		//fopen_s(&amp_file,amp_file_name, "w");
		//printf("initialized set %d\n",params_set_counter);

	bool is_next_time_step = true;							// Clac time step with Euler's method?

	// Define the output result:
# if DEBUG_FILES
	printf("open OUTPUT_FILENAME=%s\n",OUTPUT_FILENAME);
# endif
	//CBin res_bin( OUTPUT_FILENAME, BIN_WRITE );				// Creating OUTPUT binary file.

	long _step_counter = 0;			// DEBUG.
 //printf("amp_file%d= %s,set \n",params_set_counter,amp_file_name);
	//system("CLS");

	//  ---- _DEBUG_01 ---
	#if defined(__DEBUG_CMD_NOTES) || defined(__DEBUG_LOG) || defined(__DEBUG_CMD_STATUS) || defined(__DEBUG_MATLAB) || defined(__RELEASE)
		//long _step_counter = 0;			// DEBUG.
		long do_while_counter = 0;
		double prev_time_step = 0; 
	#endif
  
		int total_points = 0;
		// calculate Bassilar membrane parameters
		for (int i=0;i<SECTIONS;i++)
		{
			S_ohcd[i] = static_cast<float>(_model->_S_ohc[i]);
			S_tmd[i] = static_cast<float>(_model->_S_tm[i]);
			R_tmd[i] = static_cast<float>(_model->_R_tm[i]);
			gammad[i] = static_cast<float>(_model->configurations[params_set_counter]._gamma[i]);

			mass[i] = static_cast<float>(_model->_M_bm[i]);


			Rd[i] = static_cast<float>(_model->_R_bm[i]);
			Sd[i] = static_cast<float>(_model->_S_bm[i]);
			Qd[i] = static_cast<float>(_model->_Q[i]);
 
			rsmm[i] = 1.0f/static_cast<float>(_TriMat._m[i]);
			uu[i] = static_cast<float>(_TriMat._u[i]);
			ll[i] = static_cast<float>(_TriMat._l[i]);


		} 

		viewGPUStatus(input_params[params_set_counter].Show_Device_Data,"Pre BMOHCKernel_Init");
		// allocate GPU side arrays for BM velocity and loads BM params
		BMOHCKernel_Init(gammad, mass, rsmm, uu, ll, Rd, Sd, Qd, S_ohcd, S_tmd, R_tmd, static_cast<float>(_model->configurations[params_set_counter]._num_frequencies), static_cast<float>(_model->configurations[params_set_counter]._dbA), inputBufferTimeNodes(), bufferResultsSpeeds(), bufferLambdaSpeeds(), params_set_counter == 0,
			input_params[params_set_counter].Show_Run_Time, input_params[params_set_counter].Show_Device_Data,solver_log);
		viewGPUStatus(input_params[params_set_counter].Show_Device_Data,"Post BMOHCKernel_Init");
 //printf("kernel initialized set %d\n",params_set_counter);
	// ------------------------------------ MAIN WHILE ------------------------------------
	// as long as time is less than required run time
	int verified_points = 0;
	//printf("press any key to start\n"); getchar();
	float test_dB = START_DB;
	
	int perf_valid = 1;
# if (SIN_PERFORMENCE_CHECK_METHOD==1)
	perf_valid=0;
# endif
	//printf("input_params.duration = %f\n",input_params.duration);
	 //printf("enter main time loop set %d\n",params_set_counter);
	// debug output if requested
	if (input_params[params_set_counter].Show_Device_Data&1) {
		auto sds = showDeviceData();
		if (input_params[params_set_counter].forceMexDebugInOutput() ) {
			input_params[params_set_counter].vhout->writeVectorString("showDeviceDataVerbose", splitToVector(sds, "([^\n]+)"),0);
		} else {
			PrintFormat("%s\n", sds.c_str());
		}
		if ((input_params[params_set_counter].IS_MEX & 1) != 0) {
			input_params[params_set_counter].vhout->write_map("showDeviceData", compressDeviceData());
		}
	}
	//PrintFormat("Device data shown...\n");
	// seperate long inputs to chuncks to avoid overwhelm the GPU
	while (continueRunningInputFile(params_set_counter) || continueRunningProfileGenerator(params_set_counter,from_profile_index))
	{
		solver_log.advanceLogRound();
		remain_time = input_params[params_set_counter].duration - _current_time;
		PrintFormat("time %.3f,remained time ... %.3f\n", _current_time, remain_time);
		#if defined(__DEBUG_CMD_STATUS) || defined(__DEBUG_LOG) || defined(__RELEASE) 
		_step_counter += input_params[params_set_counter].cudaIterationsPerKernel();
		//10 if ( is_next_time_step )_step_counter += 1;

		do_while_counter = 0;
		#endif


 

  
 
	double nsample_d = 0.0; //_now->get_sample();


	long nearest = static_cast<long>( _now->_time / _now->_Ts );
	//std::cout << "nearest calculated: " << nearest << "\n";
	//_input_file->read( p_t1, nearest );
	//_input_file->read( p_t2, nearest+1 );
 

	// TODO: FIXME time_step_out should be constant , time_step should take an initial value
	float time_step = static_cast<float>(_time_step);
	float time_step_out = time_step*static_cast<float>(input_params[params_set_counter].cudaIterationsPerKernel());
	//std::cout << "time_ step: " << std::setprecision(3) << time_step << ",time_step_out: " << time_step_out << "\n";
	float tres;
	float next_time = static_cast<float>(_current_time)+(static_cast<float>((input_params[params_set_counter].Time_Blocks - 1)*input_params[params_set_counter].intervalTimeBlock()));
	float next_kernel_first_sample = next_time + input_params[params_set_counter].intervalTimeBlockOverlapped();
	
	if (input_params[params_set_counter].Perform_Memory_Test) showMemoryLeft();
	//printf("Loading %d samples start_time = %f\n",SAMPLES_BUFFER_LEN,start_time);

	// tests if start from input
	if (!input_params[params_set_counter].Run_Stage_Input()) {
		
		// no input signal is read
		if ((input_params[params_set_counter].Show_Generated_Input & 4) > 0) {
			PrintFormat("uploading calculated profiles if necessary and setting up tolerances levels\n");
		}
		if (input_params[params_set_counter].Generating_Input_Profile()) {
			// upload profiles to the GPU define powers (and frequencies if single pitch inputs) to read process run from BM velocity stage
			uploadProfiles(
				input_params[params_set_counter].inputProfile.data(),
				static_cast<int>(input_params[params_set_counter].inputProfile.size()), // number of profiles for total claculations
				from_profile_index != 0 // upload profiles array only if false
				);
			// setup convergence parameters, increase accuracy will slow the process
			setupToleranceProfile(
				input_params[params_set_counter].inputProfile.data(),
				is_first_for_param_set,
				input_params[params_set_counter].Max_M1_SP_Error_Parameter,
				input_params[params_set_counter].Max_Tolerance_Parameter,
				input_params[params_set_counter].Relative_Error_Parameters,
				input_params[params_set_counter].M1_SP_Fix_Factor,
				input_params[params_set_counter].Tolerance_Fix_Factor,
				input_params[params_set_counter].Decouple_Filter > 0 ? input_params[params_set_counter].Decouple_Filter : 1,
				from_profile_index, // profile to start calculation from
				input_params[params_set_counter].profilesPerRun(from_profile_index)
				);
		}
	} else if (input_params[params_set_counter].Generating_Input_Profile()) {
		//cout << "generating input from profile...\n";
		// updating draft white noise
		// generating white nois signal on the GPU or loading input noise signal and replicate it for each noise power level
		if (is_first_for_param_set) {
			solver_log.markTime(0);
			if (input_params[params_set_counter].JND_Signal_Source ) {
				if (input_params[params_set_counter].Has_Input_Signal()) {
					_Signal_draft = castVector<double, float>(input_params[params_set_counter].Input_Signal);
					if (_Signal_draft.size() > maxWNLength()) _Signal_draft.resize(maxWNLength());
				} else if (input_params[params_set_counter].in_file_flag) {
					_Signal_draft = input_params[params_set_counter].loadSignalFromTagToVector("Input_File", 0, static_cast<long>(input_params[params_set_counter].JND_Interval_Nodes_Full()));
					if (_Signal_draft.size() > maxWNLength()) _Signal_draft.resize(maxWNLength());
				} else {
					throw std::runtime_error("Specific JND Signal source is missing. please set either Input_Signal or Input_File");
				}
			}

			
			if (input_params[params_set_counter].JND_Noise_Source == 0) {
				copy(begin(_WN), end(_WN), begin(_WN_draft));
			} else if (input_params[params_set_counter].JND_Noise_Source == 1 ) {
				if (input_params[params_set_counter].Has_Input_Noise()) {
					_WN_draft = castVector<double, float>(input_params[params_set_counter].Input_Noise);
					if (_WN_draft.size() > maxWNLength()) _WN_draft.resize(maxWNLength());
					//std::cout << "Input noise detected, length: " << _WN_draft.size() << std::endl;
				//} else if (input_params[params_set_counter].Has_Input_Signal()) {
					//_WN_draft = castVector<double, float>(input_params[params_set_counter].Input_Signal);
				} else if ( input_params[params_set_counter].input_noise_file_flag) {
					/*
					string file_tag =  "Input_Noise_File";
					TBin<double> *signalReader = new TBin<double>(input_params[params_set_counter].vh->getString(file_tag), BIN_READ, true);
					cout << "reading " << input_params[params_set_counter].vh->getString(file_tag) << " with " << input_params[params_set_counter].JND_Interval_Nodes_Full() << " nodes\n";
					vector<double> Noise_double(input_params[params_set_counter].JND_Interval_Nodes_Full(), 0.0);
					signalReader->read_padd(Noise_double, 0, static_cast<long>(input_params[params_set_counter].JND_Interval_Nodes_Full()));
					_WN_draft = castVector<double, float>(Noise_double);
					delete signalReader;
					*/
					_WN_draft = input_params[params_set_counter].loadSignalFromTagToVector("Input_Noise_File", 0, static_cast<long>(input_params[params_set_counter].JND_Interval_Nodes_Full()));
					if (_WN_draft.size() > maxWNLength()) _WN_draft.resize(maxWNLength());
				} else {
					throw std::runtime_error("Specific JND Noise source is missing. please set either Input_Noise or Input_Noise_File");
				}
				
			}
			
			if (input_params[params_set_counter].Filter_Noise_Flag>0) {
				preProcessNoise(_WN_draft, params_set_counter);
			}
			solver_log.elapsedTimeInterrupt("WN processing", 0, 1, input_params[params_set_counter].Show_CPU_Run_Time & 2);
		}

		
		//PrintFormat("Generating Input From profile...#profiles=%d\n", input_params[params_set_counter].inputProfile.size());

		//throw std::runtime_error("Manual abort.... pre generateInputFromProfile update\n");

		// generating single pitch signals per each power level/ generate complex input signal per each power level
		generateInputFromProfile(
			input_params[params_set_counter].inputProfile.data(),
			_WN_draft.data(), // white noise array, single interval length white noise array, expected power level (linear of 1)
			static_cast<int>(_WN_draft.size()), // max length of white noise
			_Signal_draft.data(), // white noise array, single interval length white noise array, expected power level (linear of 1)
			static_cast<int>(_Signal_draft.size()), // max length of white noise
			input_params[params_set_counter].JND_Signal_Source,
			input_params[params_set_counter].Normalize_Sigma_Type,// 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
			input_params[params_set_counter].Normalize_Sigma_Type_Signal,// 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
			input_params[params_set_counter].Normalize_Noise_Energy_To_Given_Interval,
			input_params[params_set_counter].Remove_Generated_Noise_DC,
			input_params[params_set_counter].Noise_Expected_Value_Accumulating_Start_Index(), //int start_dc_expected_value_calculation,
			input_params[params_set_counter].Noise_Expected_Value_Accumulating_End_Index(), //int end_dc_expected_value_calculation,
			input_params[params_set_counter].Noise_Sigma_Accumulating_Start_Index(), //int start_dc_normalized_value_calculation,
			input_params[params_set_counter].Noise_Sigma_Accumulating_End_Index(), //int end_dc_normalized_value_calculation,
			static_cast<int>(input_params[params_set_counter].inputProfile.size()), // number of profiles for total claculations
			from_profile_index != 0, // upload profiles array only if false
			from_profile_index, // profile to start calculation from
			input_params[params_set_counter].profilesPerRun(from_profile_index),	// calculated profiles in single cuda run
			static_cast<int>(input_params[params_set_counter].calcOverlapNodes()), // for start offset
			input_params[params_set_counter].JND_Interval_Nodes_Full(), //number of nodes on actual input
			input_params[params_set_counter].JND_Interval_Nodes_Offset(),
			input_params[params_set_counter].JND_Interval_Nodes_Length(),
			static_cast<float>(input_params[params_set_counter].Fs), // sample frequency
			input_params[params_set_counter].Show_Generated_Input,
			input_samples.data(),
			input_params[params_set_counter].showGeneratedConfiguration,
			is_first_for_param_set,
			input_params[params_set_counter].Max_M1_SP_Error_Parameter,
			input_params[params_set_counter].Max_Tolerance_Parameter,
			input_params[params_set_counter].Relative_Error_Parameters,
			input_params[params_set_counter].M1_SP_Fix_Factor,
			input_params[params_set_counter].Tolerance_Fix_Factor,
			input_params[params_set_counter].Decouple_Filter > 0 ? input_params[params_set_counter].Decouple_Filter : 1,
			input_params[params_set_counter].Show_Run_Time,
			input_params[params_set_counter].Hearing_AID_FIR_Transfer_Function.data(),
			static_cast<int>(input_params[params_set_counter].Hearing_AID_FIR_Transfer_Function.size()),
			input_params[params_set_counter].Hearing_AID_IIR_Transfer_Function.data(),
			static_cast<int>(input_params[params_set_counter].Hearing_AID_IIR_Transfer_Function.size()),
			solver_log);
		PrintFormat("Generated Input From profile...\n");
		//cout << "generated input from profile...\n";
		if (input_params[params_set_counter].Perform_Memory_Test) showMemoryLeft();
	} else if (input_params[params_set_counter].sim_type==SIM_TYPE_VOICE )
	{
		//printf("Loading from file... start time =%.2f\n",start_time+input_params[params_set_counter].offset);
		//cout << "loading size is " << input_params[params_set_counter].totalTimeNodesExtendedP1() << " should be equal or larger than " << SAMPLES_BUFFER_LEN_P << "\n";
		_now->load_input_data(input_samples, __tmax(start_time + input_params[params_set_counter].offset /* - input_params[params_set_counter].overlapTime*/, 0), static_cast<int>(input_params[params_set_counter].totalTimeNodesP1()), VOICE_AMP);
		//cout << "input loaded successfully...\n";
	}
	else
	{ 
		// handle single pitch signal, ignore
		if (input_params[params_set_counter].sim_type==SIM_TYPE_PERF)
		{ 
			input_params[params_set_counter].sin_dB = test_dB;
			input_params[params_set_counter].sin_amp = CONV_dB_TO_AMP(test_dB);
			_now->gen_input_data(input_samples, __tmax(start_time + input_params[params_set_counter].offset /*- input_params[params_set_counter].overlapTime*/, 0), static_cast<int>(input_params[params_set_counter].totalTimeNodesP1()), input_params[params_set_counter].sin_freq, CONV_dB_TO_AMP(test_dB));
			
# if SIN_PERFORMENCE_TEST_2  
			input_params.sin_freq=END_FREQ+1000.0; 
# else
		if (perf_valid)
		{
			test_dB = test_dB+10.0f;
			if (test_dB>END_DB) 
			{
				test_dB = START_DB;
				input_params[params_set_counter].sin_freq+=1000.0;
			}
		}
# endif
		
# if (SIN_PERFORMENCE_CHECK_METHOD==1)
	perf_valid=1-perf_valid;
# endif
		}
		else
		{
			
			_now->gen_input_data(input_samples, __tmax(start_time + input_params[params_set_counter].offset - input_params[params_set_counter].overlapTime, 0), static_cast<int>(input_params[params_set_counter].totalTimeNodesP1()), input_params[params_set_counter].sin_freq, input_params[params_set_counter].sin_amp);
		}
	}
	start_time += input_params[params_set_counter].intervalTime();
	std::cout.precision(4);
	//std::cout << "start time updated: " << start_time << "\n";
	size_t backup_stage = input_params[params_set_counter].calculateBackupStage();
	//std::cout << "backup stage Id = " << backup_stage << std::endl;
	float time_block_length = input_params[params_set_counter].intervalTimeBlock();
	float time_block_length_overlapped = input_params[params_set_counter].intervalTimeBlockOverlapped();
	//cout << "input_params[params_set_counter].overlapTimeMicroSec= " << input_params[params_set_counter].overlapTimeMicroSec << "\n"
	//	<< "input_params[params_set_counter].totalTimeNodesP1()/ (input_params[params_set_counter].Time_Blocks+1) " << (input_params[params_set_counter].totalTimeNodesP1() / (input_params[params_set_counter].Time_Blocks+1)) << "\n";
	
	if (input_params[params_set_counter].Run_Stage_Input()) {
		viewGPUStatus(input_params[params_set_counter].Show_Device_Data, "Pre BMOHCNewKernel");
		// run BM velocity calculation, solve equations 1-10 from the Cochlear Model for Hearing Loss article
		BMOHCNewKernel(input_samples.data(), input_params[params_set_counter].Generating_Input_Profile(), static_cast<float>(_model->_w_ohc), time_step, time_step_out,
			static_cast<float>(_model->_dx), static_cast<float>(_model->_alpha_r), _model->_OHC_NL_flag, _model->_OW_flag ? 1 : 0,
			static_cast<int>(_current_time*input_params[params_set_counter].Fs), static_cast<float>(_now->_Ts),
			static_cast<float>(_model->_alpha_L), static_cast<float>(_model->_alpha_s),
			static_cast<float>(_model->_Gme), static_cast<float>(_model->_a0), static_cast<float>(_model->_a1), static_cast<float>(_model->_a2),
			static_cast<float>(_model->_sigma_ow), static_cast<float>(_model->_eta_1),
			static_cast<float>(_model->_eta_2), &tres,
			input_params[params_set_counter].Time_Blocks,
			static_cast<int>(input_params[params_set_counter].totalTimeNodesP1()),
			static_cast<int>(input_params[params_set_counter].calcOverlapNodes()),
			static_cast<long>(input_params[params_set_counter].overlapTimeMicroSec),
			1, // for the lambda full calculations
			input_params[params_set_counter].cuda_max_time_step,
			input_params[params_set_counter].cuda_min_time_step,
			input_params[params_set_counter].Decouple_Filter,
			input_params[params_set_counter].MAX_M1_SP_ERROR_Function(),
			input_params[params_set_counter].MAX_Tolerance_Function(),
			input_params[params_set_counter].Relative_Error_Parameters,
			input_params[params_set_counter].sim_type == 0, // if true will calculate max tolerance and max m1 sp error from input, should be used if input is not generated within the program	
			input_params[params_set_counter].M1_SP_Fix_Factor,
			input_params[params_set_counter].Tolerance_Fix_Factor,
			static_cast<float>(input_params[params_set_counter].SPLRefVal),
			input_params[params_set_counter].Show_Calculated_Power,
			input_params[params_set_counter].Show_Device_Data,
			input_params[params_set_counter].Show_Run_Time,
			input_params[params_set_counter].JACOBBY_Loops_Fast, // number of jcoby loops to perform on fast approximation
			input_params[params_set_counter].JACOBBY_Loops_Slow, // number of jcoby loops to perform on slow approximation
			input_params[params_set_counter].Cuda_Outern_Loops,
			input_params[params_set_counter].Run_Fast_BM_Calculation,
			input_params[params_set_counter].BMOHC_Kernel_Configuration,
			start, stop, deviceProp,solver_log); /* always calculate transient*/
		
		// log memory left if necessary
		if (input_params[params_set_counter].Perform_Memory_Test) showMemoryLeft();
		viewGPUStatus(input_params[params_set_counter].Show_Device_Data, "Post BMOHCNewKernel");
		// copy BM velocity to CPU for output (IHC compute on CPU is irrelevant)
		if (input_params[params_set_counter].Allowed_Output_Flags(0) || input_params[params_set_counter].run_ihc_on_cpu) {
			BMOHCKernel_Copy_Results(res_speeds.data(), input_params[params_set_counter].totalResultNodes(), 0);
			//PrintFormat("copied %d nodes to vector with size of %d\n", input_params[params_set_counter].totalResultNodes(), res_speeds.size());
		}
		if (input_params[params_set_counter].Convergence_Test_Flags(0)) {
			extractConvergenceTimes(convergence_times.data(), input_params[params_set_counter].totalTimeNodes());
		}
		// output convergence tests, which computations managed to achieve target accuracy
		if (input_params[params_set_counter].Convergence_Test_Flags(1)) GeneralKernel_Copy_Results_Template<float>(convergence_times_blocks.data(), getCudaConvergeSpeedBlocks(), input_params[params_set_counter].Time_Blocks);
		if (input_params[params_set_counter].Convergence_Test_Flags(2)) GeneralKernel_Copy_Results_Template<float>(convergence_loops_blocks.data(), getCudaConvergedBlocks(), input_params[params_set_counter].Time_Blocks);
		if (input_params[params_set_counter].Convergence_Test_Flags(2)) GeneralKernel_Copy_Results_Template<float>(convergence_loops.data(),getCudaConvergedTimeNodes(), input_params[params_set_counter].totalTimeNodes());
		if (input_params[params_set_counter].Convergence_Test_Flags(3)) {
			GeneralKernel_Copy_Results_Template<float>(convergence_jacoby_loops_per_iteration_blocks.data(), getCudaConvergedJacobyLoopsPerIterationBlocks(), input_params[params_set_counter].Time_Blocks);
			GeneralKernel_Copy_Results_Template<float>(convergence_jacoby_loops_per_iteration.data(), getCudaConvergedJacobyLoopsPerIteration(), input_params[params_set_counter].totalTimeNodes());
		}
		GeneralKernel_Copy_Results_Template<int>(Failed_Converged_Time_Node.data(), getCudaFailedTimeNodes(), input_params[params_set_counter].totalTimeNodes());
		GeneralKernel_Copy_Results_Template<int>(Failed_Converged_Blocks.data(), getCudaFailedBlocks(), input_params[params_set_counter].Time_Blocks);
		
	} else if (input_params[params_set_counter].Run_Stage_BM_Velocity()) {
		// if starts from BM velocity, copy to GPU to start running
		if ((input_params[params_set_counter].Show_Generated_Input & 4) > 0) {
			std::cout << "uploading results speeds due to BM Velocity stage run" << std::endl;
		}
		solver_log.markTime(2);
		input_params[params_set_counter].get_stage_data(res_speeds, 0, input_params[params_set_counter].totalTimeNodes(), SECTIONS);
		ReverseKernel_Copy_Results(res_speeds.data(), input_params[params_set_counter].totalResultNodes());
		solver_log.elapsedTimeInterrupt("Upload BM Velocity", 2, 3, input_params[params_set_counter].Show_CPU_Run_Time & 128);
	}
	
	size_t allocateTime = inputBufferTimeNodes();
	size_t writeTime = getWriteTime(input_params[params_set_counter], remain_time);
	
	// write time ensures that no more than necessary will calculate however to override default low estimates write time will ensure that it will have upwarded number for proper number of threads
	// since section multiplication already covered will ensure consecutive threads per section
	size_t calcTime = getCalcTime(writeTime);
	
	creator->setupCreator(calcTime, writeTime, allocateTime, SECTIONS, input_params, _model, out_time, is_first_for_param_set, params_set_counter, input_samples); // on next runs do not allocate memory its unnecessary
	 
	viewGPUStatus(input_params[params_set_counter].Show_Device_Data, "Post AudioGramCreator Initialization");
	size_t overlapnodes_fix_offset = input_params[params_set_counter].show_transient == 1 && (is_first_for_param_set || input_params[params_set_counter].IntervalDecoupled()) ? 0 : input_params[params_set_counter].calcOverlapNodes()*SECTIONS;
	size_t overlapnodes_fix_reduce = input_params[params_set_counter].overlapnodesFixReduce(start_time);
	if (input_params[params_set_counter].Run_Stage_Input()) {
		// output convergence tests to matlab
		int positive_decoupler = input_params[params_set_counter].Decouple_Filter > 0 ? input_params[params_set_counter].Decouple_Filter : 1;
		if (input_params[params_set_counter].Convergence_Test_Flags(0)) input_params[params_set_counter].vhout->write_vector("convergence_times", convergence_times, writeTime - (overlapnodes_fix_reduce / SECTIONS), overlapnodes_fix_offset / SECTIONS, 1);
		if (input_params[params_set_counter].Convergence_Test_Flags(1)) input_params[params_set_counter].vhout->write_vector("convergence_times_blocks", convergence_times_blocks, input_params[params_set_counter].Time_Blocks, 0, positive_decoupler);
		input_params[params_set_counter].vhout->write_vector("failed_nodes", castVector<int, float>(Failed_Converged_Time_Node), writeTime - (overlapnodes_fix_reduce / SECTIONS), overlapnodes_fix_offset / SECTIONS, 1);
		input_params[params_set_counter].vhout->write_vector("failed_blocks", castVector<int, float>(Failed_Converged_Blocks), input_params[params_set_counter].Time_Blocks, 0, positive_decoupler);
		if (input_params[params_set_counter].Convergence_Test_Flags(2)) {
			input_params[params_set_counter].vhout->write_vector("convergence_loops", convergence_loops, writeTime - (overlapnodes_fix_reduce / SECTIONS), overlapnodes_fix_offset / SECTIONS, 1);
			input_params[params_set_counter].vhout->write_vector("convergence_loops_blocks", convergence_loops_blocks, input_params[params_set_counter].Time_Blocks, 0, positive_decoupler);
		}
		if (input_params[params_set_counter].Convergence_Test_Flags(3)) {
			input_params[params_set_counter].vhout->write_vector("convergence_jacoby_loops_per_iteration", convergence_jacoby_loops_per_iteration, writeTime - (overlapnodes_fix_reduce / SECTIONS), overlapnodes_fix_offset / SECTIONS, 1);
			input_params[params_set_counter].vhout->write_vector("convergence_jacoby_loops_per_iteration_blocks", convergence_jacoby_loops_per_iteration_blocks, input_params[params_set_counter].Time_Blocks, 0, positive_decoupler);
		}
	}
	 // output velocity to matlab 
	 if (input_params[params_set_counter].Allowed_Output_Flags(0)) {
		 solver_log.markTime(4);
		 //_output_file.append_buffer(res_speeds, writeTime*SECTIONS - overlapnodes_fix_reduce, overlapnodes_fix_offset);
		 int copied_nodes = writeTime*SECTIONS - overlapnodes_fix_reduce;
		 input_params[params_set_counter].vhout->write_vector("output_results", res_speeds, copied_nodes, overlapnodes_fix_offset, SECTIONS);
		 //PrintFormat("copied %d nodes from vector with size of %d\n", copied_nodes, res_speeds.size());
		 solver_log.elapsedTimeInterrupt("Save BM Velocity", 4, 5, input_params[params_set_counter].Show_CPU_Run_Time & 8);
	 }
	 if (input_params[params_set_counter].Perform_Memory_Test) showMemoryLeft();
	 if (input_params[params_set_counter].saveLambda() || backup_stage > -1) {
		 allocateBuffer((input_params[params_set_counter].allocateFullBuffer() ? LAMBDA_COUNT : 1)*calcTime*SECTIONS, input_params[params_set_counter].Show_Run_Time, start, stop, deviceProp);
	 }

	 if ( !input_params[params_set_counter].disable_lambda ) {
		 creator->Failed_Converged_Blocks = Failed_Converged_Blocks;
		 creator->runInCuda(res_speeds.data(),solver_log);
		 // runs AN response on CPU, this algorithm is outdated
		 if (!creator->isRunIncuda()) {
			std::cout << "Run Lambda on CPU" << std::endl;
			 creator->loadSavedSpeeds(res_speeds);
			 creator->ac_filter();
			 creator->filterDC();
			 creator->calcLambdas(solver_log);
		 }
		 viewGPUStatus(input_params[params_set_counter].Show_Device_Data,"Post Lambda Calculation");
		 //std::cout << "saving lambda now...\n";
		 solver_log.markTime(6);
		 // output lambdas to matlab if necessary
		 creator->saveLambdas( overlapnodes_fix_reduce, overlapnodes_fix_offset, solver_log);
		 solver_log.elapsedTimeInterrupt("Save lambda", 6, 7, input_params[params_set_counter].Show_CPU_Run_Time & 8);
		// calculating JND
		 if (input_params[params_set_counter].Calculate_JND  && static_cast<float>(input_params[params_set_counter].duration - next_time) < input_params[params_set_counter].JND_Interval_Duration) {
			 PrintFormat("Saving JND...\n");
			 solver_log.markTime(8);
			 // here JND output is occur, note that results built as 2D arrays for matlab, seperated by diffrent inputs (each is different combination of pitvh and noise)
			 if (input_params[params_set_counter].isWritingRawJND()) {
				 int bsize = (input_params[params_set_counter].JND_Calculate_AI() ? 1 : 0) + (input_params[params_set_counter].JND_Calculate_RMS() ? 1 : 0);
				 bsize = bsize*input_params[params_set_counter].numberOfJNDIntervals();
				 if (bsize > JND_MAX_BUFFER_SIZE) bsize = JND_MAX_BUFFER_SIZE;
				 //OutputBuffer<float> jnd_file(input_params[params_set_counter].Raw_JND_Target_File(), bsize, false);
				 int num_of_calculated = input_params[params_set_counter].numberOfJNDIntervals() - static_cast<int>(input_params[params_set_counter].JND_Reference_Intervals_Positions.size());
				 creator->Failed_Converged_Signals.resize(num_of_calculated);
				 size_t rmsMsize = 1;
				 if (input_params[params_set_counter].JND_Column_Size() > 0)  rmsMsize = num_of_calculated / (input_params[params_set_counter].JND_Column_Size()*input_params[params_set_counter].JNDInputTypes());
				 size_t Mcolumns_size = rmsMsize * (input_params[params_set_counter].JND_Signal_Source ? 1 : input_params[params_set_counter].JND_Column_Size());
				// PrintFormat("Record %lu failed signals\n", creator->Failed_Converged_Signals.size());
				 input_params[params_set_counter].vhout->write_vector("failed_signals", castVector<int, float>(creator->Failed_Converged_Signals), num_of_calculated, 0, Mcolumns_size);
				if (input_params[params_set_counter].JND_Calculate_AI() ) {
					 vector<float> jndBuffer = castVector<double, float>(creator->AiJNDall);
					 jndBuffer.resize(num_of_calculated);
					 
					 if (input_params[params_set_counter].JND_Signal_Source == 0) {
						// PrintFormat("Record %lu AI power\n", jndBuffer.size());
						 input_params[params_set_counter].vhout->write_vector("jnd_ai_power", jndBuffer, num_of_calculated, 0, num_of_calculated/ input_params[params_set_counter].JND_Column_Size());

					 }
					 jndBuffer = transposeVector(jndBuffer, Mcolumns_size);
					 input_params[params_set_counter].vhout->write_vector("jnd_ai", jndBuffer, num_of_calculated, 0, Mcolumns_size);
					 if ((input_params[params_set_counter].JND_Include_Legend & 4) != 0) {
						// PrintFormat("Record AI legend\n");
						 input_params[params_set_counter].vhout->writeVectorString("jnd_ai_legend", transposeVector(creator->getSimpleLegends(0, input_params[params_set_counter].JND_Calculated_Intervals_Positions), rmsMsize), rmsMsize);
					 }
				 }
				 if (input_params[params_set_counter].JND_Calculate_RMS() ) {
					 vector<float> jndBuffer = castVector<double, float>(creator->RateJNDall);
					 jndBuffer.resize(num_of_calculated);
					 if (input_params[params_set_counter].JND_Signal_Source == 0) {
						// PrintFormat("Record %lu RMS power\n", jndBuffer.size());
						 input_params[params_set_counter].vhout->write_vector("jnd_rms_power", jndBuffer, num_of_calculated, 0, num_of_calculated / input_params[params_set_counter].JND_Column_Size());

					 }
					 jndBuffer = transposeVector(jndBuffer, Mcolumns_size);
					 //jnd_file.append_buffer(jndBuffer, num_of_calculated, 0);
					 //PrintFormat("Record %lu RMS buffer\n", jndBuffer.size());
					 input_params[params_set_counter].vhout->write_vector("jnd_rms", jndBuffer, num_of_calculated, 0, Mcolumns_size);
					 if ((input_params[params_set_counter].JND_Include_Legend & 2) != 0) {
						// PrintFormat("Record RMS legend\n");
						 input_params[params_set_counter].vhout->writeVectorString("jnd_rms_legend", transposeVector(creator->getSimpleLegends(1, input_params[params_set_counter].JND_Calculated_Intervals_Positions), rmsMsize), rmsMsize);
					 }
				 }
				 //jnd_file.flush_buffer();
				 if (input_params[params_set_counter].vhout->Is_Matlab_Formatted() == 0) {
					 input_params[params_set_counter].vhout->Flush_Major(input_params[params_set_counter].Raw_JND_Target_File());
					 input_params[params_set_counter].vhout->removeMajor(input_params[params_set_counter].Raw_JND_Target_File(), 1);
				 }
			 }
			 if (input_params[params_set_counter].sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS ) {
				 creator->calcComplexJND(input_params[params_set_counter].Calculate_From_Mean_Rate?creator->RateJNDall:creator->AiJNDall);
				 //OutputBuffer<float> jnd_file_final(input_params[params_set_counter].JND_File_Name, JND_MAX_BUFFER_SIZE, false);
				 vector<float> jndBuffer = castVector<double, float>(creator->ApproximatedJNDall);
				 vector<float> jndBufferWarnings = castVector<int, float>(creator->ApproximatedJNDallWarnings);
				 vector<float> jndFailedSummariesWarnings = castVector<int, float>(creator->Failed_Converged_Summaries);
				// PrintFormat("Record final\n");
				 input_params[params_set_counter].vhout->write_vector("jnd_final", jndBuffer, input_params[params_set_counter].numberOfApproximatedJNDs(), 0, input_params[params_set_counter].JNDInputTypes());
				// PrintFormat("Record final warnings\n");
				 input_params[params_set_counter].vhout->write_vector("jnd_final_warnings", jndBufferWarnings, input_params[params_set_counter].numberOfApproximatedJNDs(), 0, input_params[params_set_counter].JNDInputTypes());
				// PrintFormat("Record final conv warnings\n");
				 input_params[params_set_counter].vhout->write_vector("jnd_final_signal_convergence_warning", jndFailedSummariesWarnings, input_params[params_set_counter].numberOfApproximatedJNDs(), 0, input_params[params_set_counter].JNDInputTypes());
				 if ((input_params[params_set_counter].JND_Include_Legend & 1) != 0) {
					 std::vector<std::string> final_legends = creator->getLegends(1);
					// PrintFormat("Record final legend #%lu, M=%lu\n", final_legends.size(), input_params[params_set_counter].targetJNDNoises());
					 input_params[params_set_counter].vhout->writeVectorString("jnd_final_legend", final_legends, input_params[params_set_counter].targetJNDNoises());
					// PrintFormat("Record passed final legend #%lu, M=%lu\n", final_legends.size(), input_params[params_set_counter].targetJNDNoises());
				 }
				 //jnd_file_final.append_buffer(jndBuffer, input_params[params_set_counter].numberOfApproximatedJNDs(), 0);
				 //jnd_file_final.flush_buffer();
				 if (input_params[params_set_counter].vhout->Is_Matlab_Formatted() == 0) {
					// PrintFormat("Record final in flush\n");
					 input_params[params_set_counter].vhout->Flush_Major(input_params[params_set_counter].JND_File_Name);
					 input_params[params_set_counter].vhout->removeMajor(input_params[params_set_counter].JND_File_Name, 1);
				 }
				 //PrintFormat("Record final post flush\n");
			 }

			 
			 PrintFormat("Saving JND complete\n");
			 solver_log.elapsedTimeInterrupt("Save JND", 8, 9, input_params[params_set_counter].Show_CPU_Run_Time & 16);
			 //std::cout << "Saved JND..." << std::endl;
		 }
		
		 releaseBuffer( input_params[params_set_counter].Show_Run_Time, start, stop, deviceProp);
		 //Output Generated Input signal if requested
		 if (input_params[params_set_counter].Show_Generated_Input&1) {
			 int allNodes = input_params[params_set_counter].profilesPerRun(from_profile_index)*input_params[params_set_counter].JND_Interval_Nodes_Full();
			 //cout << "nodes up to here " << allNodes << " need to be less than inputBufferTimeNodes() = " << inputBufferTimeNodes() << ",from_profile_index = "<<from_profile_index<<"\n";
			 if (input_params[params_set_counter].IS_MEX) {
				 PrintFormat("Saving Input\n");
				 input_params[params_set_counter].vhout->write_vector("Input", input_samples, allNodes,0, 1);
				 PrintFormat("Saving Input complete\n");
			 } else if (input_params[params_set_counter].in_file_flag == 1) {
				 creator->saveFloatArrayToDisk(input_samples, allNodes, allNodes, input_params[params_set_counter].in_file_name, false, from_profile_index == 0);
			 } else {
				 cerr << "Input file parameter is missing, cannot write generated input samples...\n";
			 }

		 }


		 if (input_params[params_set_counter].Generating_Input_Profile()) {
			 from_profile_index += input_params[params_set_counter].profilesPerRun(from_profile_index);
		 }
		 
	 }
	 is_first = false; // memory already allocated
	 is_first_for_param_set = false;
 
	first_max_iter = 0;
	out_time += input_params[params_set_counter].totalTimeNodes() * _now->_Ts;
 
	total_points += input_params[params_set_counter].totalTimeNodes();
  
	//cout << "out time " << out_time  << "\n";
		
	int loop_current_times = 0;
		for (int check=0;_current_time<start_time;check++) {
 
		_current_time += _now->_Ts; //(input_params[params_set_counter].cudaIterationsPerKernel()*_time_step); 
		
             _now->_time_step=_time_step;//9c
			 loop_current_times++;
			
			//  ---- For the RELEASE version, Oded ---
			#ifdef __RELEASE
 
# define _STEP_COUNTER_OFF _CUDA_ITERATIONS_PER_KERNEL 


			#endif

		} // for (check ...) is_next_time..
		//cout << "loop runs are: " << loop_current_times << "\n";
		solver_log.markTime(11);
		if (loop_current_times > 0) {
			if (loop_current_times == 1) {
				_past->copy_state(_now);
			}
			else {
				_past->restart_state(_current_time - _now->_Ts, _time_step);
			}
			_now->restart_state(_current_time, _time_step);
		}
		solver_log.elapsedTimeInterrupt("State update", 11, 12, input_params[params_set_counter].Show_CPU_Run_Time & 4);
		
		if ( input_params[params_set_counter].continues == 0 ) {
			break;
		}

	}	// while
	
	solver_log.markTime(6);
	solver_log.elapsedTimeInterrupt("Flush to output file", 6, 7, input_params[params_set_counter].Show_CPU_Run_Time & 4);

	std::vector<float> timers_marked = solver_log.getTimers();
	// put different log parmeters to output
	input_params[params_set_counter].vhout->write_vector("timers_marked", timers_marked, timers_marked.size(), 0, solver_log.getFlagsPerRound());
	if (input_params[params_set_counter].forceMexDebugInOutput()) {
		std::ostringstream osslog("");
		osslog << "solver_log_param_set_" << params_set_counter << "_" << start_time;
		std::string sslogname = osslog.str();
		std::replace(sslogname.begin(), sslogname.end(), '.', '_');
		//std::cout << "flushing to " << sslogname << std::endl;
		input_params[0].vhout->flushToIOHandler(solver_log, sslogname);
	}
	else {
		solver_log.flushLog();
	}
	//
	// ------------------------------------ MAIN WHILE ENDS ------------------------------------


		#ifdef __DEBUG_MATLAB	// MATLAB debug mode - Ploting Final Results
		//Plot_On_Screen( _now, _model, _step_counter );		
		#endif
	// arrange output memory
	if ((input_params[params_set_counter].IS_MEX & 1) == 0) {
		input_params[params_set_counter].vhout->Flush_Major(input_params[params_set_counter].output_name);
		input_params[params_set_counter].vhout->removeMajor(input_params[params_set_counter].output_name, 1);
	}
	}
	
# if PRINT_FULL_OUTPUTS
	//fclose(res_file);
# endif
	
# if SIN_PERFORMENCE_TEST_
		fclose(fperf);
		fclose(fperf_mat);
# endif

		if (input_params[0].forceMexDebugInOutput()) {

			
			input_params[0].vhout->flushToIOHandler(solver_log, "solver_log");
		}
		else {

			solver_log.flushLog();

		}
}	// --- End of CSolve::Run_Cuda() ---

void CSolver::clearSolver() {
	// this clear all dynamic memory allocations and log times
	solver_log.markTime(6);
	viewGPUStatus(input_params[0].Show_Device_Data, "Pre BMOHCKernel_Free");
	BMOHCKernel_Free();
	InputProfilesArrayTermination();
	viewGPUStatus(input_params[0].Show_Device_Data, "Post BMOHCKernel_Free");
	if (input_params[0].Show_CPU_Run_Time & 32) {
		PrintFormat("Pre delete creator\n");
	}
	delete creator;
	if (input_params[0].Show_CPU_Run_Time & 32) {
		PrintFormat("Post delete creator\n");
	}
	solver_log.markTime(7);
	if (input_params[0].Show_CPU_Run_Time & 32) {
		PrintFormat("Pre additional message\n");
		PrintFormat("AudioLab Delete Memory allocations: %ld (msec)\n", solver_log.getElapsedTime(6, 7));
	}
	
}

bool CSolver::reloadBufferFromFile(int params_set_counter) {
	return !input_params[params_set_counter].Has_Input_Signal()
		&& static_cast<size_t>((_current_time + input_params[params_set_counter].intervalTime())*input_params[params_set_counter].Fs) > static_cast<size_t>(_buffer_start_position) +_input_buffer.size()
		&& static_cast<size_t>((_run_time - 1)*input_params[params_set_counter].Fs) > static_cast<size_t>(_buffer_start_position) + _input_buffer.size();
}



// Open all data files for work:
void CSolver::Open_Data_Files(int params_set_counter)
{
	//cout << boost::format("Opening Files -> %s\n") % input_params[params_set_counter].in_file_name;
	if (input_params[params_set_counter].sim_type==SIM_TYPE_VOICE)
	{

		if (input_params[params_set_counter].Has_Input_Signal()) {
			_input_buffer = vector<double>(input_params[params_set_counter].Input_Signal); // Create OUTPUT file
		} else 	if (input_params[params_set_counter].in_file_flag)
		{		
			cout << "Openning file " << input_params[params_set_counter].in_file_name << "\n";
			_input_file = new TBin<double>(input_params[params_set_counter].in_file_name, BIN_READ);
			
			//cout << boost::format("file openned %s\n") % input_params[params_set_counter].in_file_name;
			_buffer_start_position = static_cast<long>(_current_time*input_params[params_set_counter].Fs);
			long buffer_found_length = __tmin(INPUT_MAX_SIZE, static_cast<long>(_input_file->_file_length - _buffer_start_position));
			_input_buffer.resize(buffer_found_length);
			_input_file->read_padd(_input_buffer, _buffer_start_position, buffer_found_length);
		 
		} 
		
		
	}
	

}


// Save calculated (and desired) data into the harddisk: this function is outdated
void CSolver::Save_Results( const long step_counter, const double current_time  )
{
	
	//if  ( !(step_counter % SAVE_RATE) && (current_time <= SAVE_START_TIME) )		
	//	return;

	if (current_time<next_save_time) return;
	next_save_time += _now->_Ts;

	//printf("Saving output: time = %e\n",_now->_time); getchar();

	
}	// --- End of CSolve::Save_Results() ---


// Close all open data files (input and output): used in old output data to disk mode
void CSolver::Close_Data_Files(int params_set_counter)
{
	if (input_params[params_set_counter].sim_type == SIM_TYPE_VOICE && _input_file!=NULL&& _input_file->_is_opened())
	{
		_input_file->close_file();
		delete _input_file;
		_input_file = NULL;
	}

	

}
// summarize number of cuda block to execute each time
size_t CSolver::maxTimeBlocks() {
	size_t mtb = 0;
	for (int i = 0; i < num_params; i++) {
		if (mtb < input_params[i].Time_Blocks)	mtb = input_params[i].Time_Blocks;
	}
	return mtb;
}

// summerize array size output length for time block in cuda to ensure enough memory is allocated
size_t CSolver::longestTimeBlock() {
	size_t mtb = 0;
	for (int i = 0; i < num_params; i++) {
		if (mtb < input_params[i].calcTimeBlockNodes())	mtb = input_params[i].calcTimeBlockNodes();
	}
	return mtb;
}
// length of input buffer to allocate
size_t CSolver::inputBufferTimeNodes() {
	return longestTimeBlock()*(maxTimeBlocks() + 1) + 2 * bufferOverlapNodes();
}

// length of output buffer to allocate
size_t CSolver::bufferResultsSpeeds() {
	size_t brs = inputBufferTimeNodes()*SECTIONS;
	//cout << " allocated maximum of " << brs << " nodes for result in bufferResultsSpeeds()\n";
	return brs;
}

// length of lambda buffer to allocate
size_t CSolver::bufferLambdaSpeeds() {
	return bufferResultsSpeeds()*LAMBDA_COUNT;
}

// calculate output buffer overlapped nodes to calculate final result length
size_t CSolver::bufferOverlapNodes() {
	size_t bon = 0;
	for (int i = 0; i < num_params; i++) {
		if (bon < input_params[i].calcOverlapNodes())	bon = input_params[i].calcOverlapNodes();
	}
	return bon;
}

