# include "params.h"

CParams::CParams(void) {
	sim_type = SIM_TYPE_SIN; 
	sin_freq = INPUT_SIG_FREQ;
	sin_dB = INPUT_SIGNAL_dB;
	sin_amp = INPUT_SIG_AMP; 
	sin_freq = DEFAULT_TEST_FREQ;
	in_file_flag = 0;
	input_noise_file_flag = 0;
	Run_Stage_Calculation = 0;
	SPLRefVal = SPLRef;
	num_frequencies = 1;
	disable_lambda = false;
	hasJND_Raw_Output = false;
	Allowed_Outputs = 0;
	gamma_file_flag = 0;
	database_creation = 0;
	Normalize_Sigma_Type = 0;
	Normalize_Sigma_Type_Signal = 0;
	show_transient = 0; // default will not show transient(first overlap period
	disable_advanced_memory_handling = false;
	run_ihc_on_cpu = false;
	Show_JND_Configuration = false;
	Decouple_Filter = 0;
	nerve_file_flag =0;
	in_file_name[0] = 0;
	gamma_file_name[0] = 0;
	strcpy_s(backup_file_name,SAVED_SPEEDS_FILE);
	duration = FULL_TIME_WRITE; // TOTAL_DEFAULT_RUN_TIME;
	continues =1;
	offset = 0;
	Show_Run_Time = 0;
	frequncies_analysis = 0;
	loadPrevData = 0;
	sprintf_s<FILE_NAME_LENGTH_MAX>(lambda_high_file_name, LAMBDA_PATH, HIGH_LAMBDA_FILE);
	sprintf_s<FILE_NAME_LENGTH_MAX>(lambda_medium_file_name, LAMBDA_PATH, MEDIUM_LAMBDA_FILE);
	sprintf_s<FILE_NAME_LENGTH_MAX>(lambda_low_file_name, LAMBDA_PATH, LOW_LAMBDA_FILE);

	// default values of lambda files names saved for debugging
	sprintf_s<FILE_NAME_LENGTH_MAX>(default_lambda_high_file_name, LAMBDA_PATH, HIGH_LAMBDA_FILE);
	sprintf_s<FILE_NAME_LENGTH_MAX>(default_lambda_medium_file_name, LAMBDA_PATH, MEDIUM_LAMBDA_FILE);
	sprintf_s<FILE_NAME_LENGTH_MAX>(default_lambda_low_file_name, LAMBDA_PATH, LOW_LAMBDA_FILE);
	
	sprintf_s<FILE_NAME_LENGTH_MAX>(output_file_name, OUTPUT_RESULTS_FILE);
	sprintf_s<FILE_NAME_LENGTH_MAX>(ac_filter_file_name, AC_TIME_FILTER_FILE);
	ihc_vector = std::vector<float>(SECTIONS, HEALTY_IHC);
	ohc_vector = std::vector<float>(SECTIONS, HEALTY_OHC);
	ihc_mode = 0;
	ohc_mode=0;
	Lambda_SAT = RSAT;
	// work buffer parameters
	Fs = SAMPLE_RATE;
	Time_Blocks = TIME_SECTIONS;
	Time_Block_Length = SAMPLE_BUFFER_LEN_SHORT_INTERVAL;
	overlapTimeMicroSec = TIME_OFFSET;
	overlapTime = toSeconds(overlapTimeMicroSec);
	Processed_Interval = Time_Blocks*Time_Block_Length;
	// prime filter function params
	ac_filter_mode = 1; // 0 for file, 1 for function
	Fc = 800.0f; // Frequency cut for lpf
	Fpass = 600.0f; // frequency pass band for the filter for lpf
	Fstop = 1600.0f; // stop band for lpf
	Apass = 3; // gain on fpass will be 3db+filter gain for lpf
	Astop = 60; // Attentunation stop of the filetr for lpf
	Wpass = 1; // weight of passband in equiripple
	Wstop = 100; // weight of stop in equiripple
	FilterOrder = -1;
	Slope = 5;
	scaleBMVelocityForLambdaCalculation = Scale_BM_Velocity_For_Lambda_Calculation;
	Tdelta = Tdc;
	eta_AC = AMP_AC;
	eta_DC = AMP_DC;
	cuda_max_time_step = CUDA_MAX_TIME_STEP;
	cuda_min_time_step = CUDA_MIN_TIME_STEP;
	mode = "";
	butterType = PASSBAND;
	filterName = "EquiRipple"; // filter name like butterworth or equiripple, should be lpf
	show_filter = false; // if true show generated filter
	deltasFound = false; // if true will calculate weights from Apass, Astop if constrained mode active than will use default Apass, Astop
	minOrder = false; // if true will override order and use Oppenheim & Schafer 2nd addition DSP formula 7.104 to calculate mininmum order


	Max_M1_SP_Error_Parameter = MAX_M1_SP_ERROR;
	Max_Tolerance_Parameter = MAX_TOLERANCE;
	Relative_Error_Parameters = 0; // default use absolute values	 
	Tolerance_Fix_Factor = DEFAULT_Tolerance_Fix_Factor;
	M1_SP_Fix_Factor = DEFAULT_M1_SP_Fix_Factor;
	Show_Calculated_Power = 0;
	// JND parameters
	Calculate_JND = false;
	Calculate_JND_Types = (1 << 16) - 1; // all possible flags by default
	JND_Ignore_Tail_Output = static_cast<float>(overlapTime);
	JND_Interval_Duration = JND_INTERVAL_SIZE;
	JND_Noise_Source = 0; // default self generate
	JND_Reference_Intervals_Positions = std::vector<int>();
	JND_Interval_To_Reference = std::vector<int>();
	JND_Calculated_Intervals_Positions = std::vector<int>();
	JND_WN_Additional_Intervals = std::vector<float>();
	Calculate_From_Mean_Rate = true;
	Examine_Configurations = 0;
	JND_Interval_Head = JND_BLOCK_HEAD;
	JND_Interval_Tail = JND_BLOCK_TAIL;
	JACOBBY_Loops_Fast = _CUDA_JACOBBY_LOOPS2;
	JACOBBY_Loops_Slow = _CUDA_JACOBBY_LOOPS1;
	Cuda_Outern_Loops = _CUDA_OUTERN_LOOPS;
	JND_File_Name = "";
	JND_Raw_File_Name = "";
	Type_TEST_Output = "MeanRate";
	Convergence_Tests = 0;
	M_tot = std::vector<float>(SECTIONS,static_cast<float>(AN_FIBERS)/static_cast<float>(SECTIONS));
	View_Complex_JND_Source_Values = false;
	Perform_Memory_Test = false;
	Calculate_JND_On_CPU = false;
	JND_Delta_Alpha_Time_Factor = Const_JND_Delta_Alpha_Time_Factor;
	// debug flags
	Review_Lambda_Memory = false;
	Review_AudioGramCall = false;
	generatedInputProfile = false;
	JND_Include_Legend = 0;
	Filter_Noise_Flag = 0;
	Generate_One_Ref_Per_Noise_Level = true;
	Aihc = std::vector<float>(LAMBDA_COUNT);
	Aihc[0] = static_cast<float>(HIGH_FREQ_NERVE);
	Aihc[1] = static_cast<float>(MEDIUM_FREQ_NERVE);
	Aihc[2] = static_cast<float>(LOW_FREQ_NERVE);
	Complex_Profile_Noise_Level_Fix_Factor = std::vector<double>(1,0.0);
	Complex_Profile_Noise_Level_Fix_Addition = std::vector<double>(1, 0.0);
	Complex_Profile_Power_Level_Divisor = std::vector<double>(1, 0.0);
	W = std::vector<float>(LAMBDA_COUNT);
	W[0] = static_cast<float>(FIBER_HIGH_DIST);
	W[1] = static_cast<float>(FIBER_MEDIUM_DIST);
	W[2] = static_cast<float>(FIBER_LOW_DIST);

	spontRate = std::vector<float>(LAMBDA_COUNT);
	spontRate[0] = static_cast<float>(SPONT_HIGH_RATE);
	spontRate[1] = static_cast<float>(SPONT_MEDIUM_RATE);
	spontRate[2] = static_cast<float>(SPONT_LOW_RATE);

	// complex input generation patterns   default assignments
	float a[] = { 250.0f, 500.0f, 1000.0f, 2000.0f, 3000.0f, 4000.0f, 6000.0f};
	testedFrequencies = std::vector<float>(a,a+7);
	showGeneratedConfiguration = false;
	testedNoises = std::vector<float>(1,1111.0f);
	testedPowerLevels = std::vector<float>(1,10.0f);
	Noise_Expected_Value_Accumulating_Boundaries = std::vector<double>(2, 0.0);
	Noise_Sigma_Accumulating_Boundaries = std::vector<double>(2, 0.0);
	JND_USE_Spont_Base_Reference_Lambda = 1;
	Show_Generated_Input = 0;
	maxdBSPLJND = JND_Sat_Value_DB; // max value in dB for jnd value to prevent infinities
	Approximated_JND_EPS_Diff = 1.0f;
	Remove_Generated_Noise_DC = 0.0;
	Normalize_Noise_Energy_To_Given_Interval = 0.16;
	Discard_BM_Velocity_Output = false;
	Discard_Lambdas_Output = false;
	Remove_Transient_Tail = 0;
	Run_Fast_BM_Calculation = 0;
	Show_CPU_Run_Time = 0;
	Complex_JND_Calculation_Types = 1;
	Verbose_BM_Vectors = std::vector<std::string>(0);
	IS_MEX = 0;
	Mex_Debug_In_Output = 0;
	// debug stages

	addStageNumber("AvgMeanRate",1);
	addStageNumber("dMeanRate",2);
	addStageNumber("dLambda",3);
	for (int ind = 0; ind < LAMBDA_COUNT; ind++) {
		std::ostringstream oss("");
		oss << "dLambda(" << ind << ")";
		std::string str = oss.str();
		addStageNumber(str,3);
		std::ostringstream oss1("");
		oss1 << "JND_Lambda(" << ind << ")";
		std::string str1 = oss1.str();
		addStageNumber(str1,8);
	}
	addStageNumber("preFisherAI",4);
	addStageNumber("preFisherAITimeReduced",5);
	addStageNumber("CRLB_RA",6);

	addStageNumber("AC",9);
	addStageNumber("DC",10);
	addStageNumber("dS",11);
	addStageNumber("Shigh",12);
	addStageNumber("IHC",13);
	addStageNumber("PRE_IHC",14);
	BMOHC_Kernel_Configuration = 2;
	BMOHC_Kernel_Configurations_Names.push_back("cudaFuncCachePreferNone");
	BMOHC_Kernel_Configurations_Names.push_back("cudaFuncCachePreferShared");
	BMOHC_Kernel_Configurations_Names.push_back("cudaFuncCachePreferL1 - Default Value");
	BMOHC_Kernel_Configurations_Names.push_back("cudaFuncCachePreferEqual");
	vh = NULL;
	vhout = NULL;
	Decouple_Unified_IHC_Factor = 0;
	Hearing_AID_FIR_Transfer_Function = std::vector<float>(1, 1.0f);
	Hearing_AID_IIR_Transfer_Function = std::vector<float>(1, 1.0f);
}
CParams::CParams(unsigned int mex_status) :CParams(){
	IS_MEX = mex_status;

}
CParams::~CParams(void) {
	std::cout << "terminating params " << std::endl;
	if (vhout != NULL) {
		delete vhout;
		vhout = NULL;
	}
	std::cout << "terminated output " << std::endl;
	if (vh != NULL) {
		delete vh;
		vh = NULL;
	}
	std::cout << "terminated input " << std::endl;
	//delete vh;
}
CParams& CParams::operator=(const CParams& p) { 
	sim_type = p.sim_type;
	sin_freq = p.sin_freq;
	sin_dB = p.sin_dB;
	sin_amp = p.sin_amp;
	SPLRefVal = p.SPLRefVal;
	database_creation = p.database_creation;
	in_file_flag = p.in_file_flag;
	gamma_file_flag = p.gamma_file_flag;
	nerve_file_flag = p.nerve_file_flag;
	strcpy_s(nerve_file_name,p.nerve_file_name);
	duration = p.duration;
	continues = p.continues;
	Lambda_SAT = p.Lambda_SAT;
	Allowed_Outputs = p.Allowed_Outputs;
	Tdelta = p.Tdelta;
	eta_AC = p.eta_AC;
	eta_DC = p.eta_DC;
	scaleBMVelocityForLambdaCalculation = p.scaleBMVelocityForLambdaCalculation;
	Fs = p.Fs;
	Mex_Debug_In_Output = p.Mex_Debug_In_Output;
	Decouple_Unified_IHC_Factor = p.Decouple_Unified_IHC_Factor;
	JND_Include_Legend = p.JND_Include_Legend;
	JACOBBY_Loops_Fast = p.JACOBBY_Loops_Fast;
	JACOBBY_Loops_Slow = p.JACOBBY_Loops_Slow;
	Time_Blocks = p.Time_Blocks;
	Time_Block_Length = p.Time_Block_Length;
	run_ihc_on_cpu = p.run_ihc_on_cpu;
	disable_advanced_memory_handling = p.disable_advanced_memory_handling;
	disable_lambda = p.disable_lambda;
	offset = p.offset;
	ihc_vector = p.ihc_vector;
	ohc_vector = p.ohc_vector;
	ihc_mode = p.ihc_mode;
	ohc_mode=  p.ohc_mode;
	num_frequencies = p.num_frequencies;
	loadPrevData = p.loadPrevData;
	Decouple_Filter = p.Decouple_Filter;
	vh = p.vh;
	Max_M1_SP_Error_Parameter = p.Max_M1_SP_Error_Parameter;
	Max_Tolerance_Parameter = p.Max_Tolerance_Parameter;
	Relative_Error_Parameters = p.Relative_Error_Parameters; // default use absolute values
	Tolerance_Fix_Factor = p.Tolerance_Fix_Factor;
	M1_SP_Fix_Factor = p.M1_SP_Fix_Factor;
	Show_Calculated_Power = p.Show_Calculated_Power;
	for(int i=0;i<num_frequencies;i++) {
		frequncies[i] = p.frequncies[i];
	}
	frequncies_analysis = p.frequncies_analysis;
	strcpy_s(in_file_name, p.in_file_name);
	strcpy_s(in_noise_file_name, p.in_noise_file_name);
	strcpy_s(gamma_file_name, p.gamma_file_name);
	strcpy_s(database_file_name,p.database_file_name);
	strcpy_s(backup_file_name,p.backup_file_name);
	strcpy_s(output_file_name, p.output_file_name);
	strcpy_s(lambda_high_file_name, p.lambda_high_file_name);
	strcpy_s(lambda_medium_file_name,p.lambda_medium_file_name);
	strcpy_s(lambda_low_file_name,p.lambda_low_file_name);
	strcpy_s(ac_filter_file_name,p.ac_filter_file_name);

	show_transient = p.show_transient;
	// prime filter function params
	ac_filter_mode = p.ac_filter_mode; // 0 for file, 1 for function
	Fc = p.Fc; // Frequency cut for lpf
	Fpass = p.Fpass; // frequency pass band for the filter for lpf
	Fstop = p.Fstop; // stop band for lpf
	Apass = p.Apass; // gain on fpass will be 3db+filter gain for lpf
	Astop = p.Astop; // Attentunation stop of the filetr for lpf
	Slope = p.Slope;
	Wpass = p.Wpass;
	Wstop = p.Wstop;
	
	cuda_max_time_step = p.cuda_max_time_step;
	cuda_min_time_step = p.cuda_min_time_step;
	overlapTimeMicroSec = p.overlapTimeMicroSec;
	FilterOrder = p.FilterOrder;
	mode.assign(p.mode);
	filterName.assign(p.filterName); // filter name like butterworth or equiripple, should be lpf
	Type_TEST_Output.assign(p.Type_TEST_Output);
	show_filter = p.show_filter; // if true show generated filter
	deltasFound = p.deltasFound;
	minOrder = p.minOrder;
	butterType = p.butterType;
	filtersMap = p.filtersMap;
	filtersMapRaw = p.filtersMapRaw;
	filtersKeysStat = p.filtersKeysStat;
	overlapTime = p.overlapTime;
	Processed_Interval = p.Processed_Interval;
	Time_Block_Length_Microseconds = p.Time_Block_Length_Microseconds;
	Aihc = std::vector<float>(p.Aihc);
	spontRate = std::vector<float>(p.spontRate);
	Complex_JND_Calculation_Types = p.Complex_JND_Calculation_Types;
	Run_Stage_Calculation = p.Run_Stage_Calculation;
	// JND parameters	Max_M1_SP_Error_Parameter = MAX_M1_SP_ERROR;
	Max_Tolerance_Parameter = p.Max_Tolerance_Parameter;
	
	Calculate_JND = p.Calculate_JND;
	View_Complex_JND_Source_Values = p.View_Complex_JND_Source_Values;
	JND_Interval_Duration = p.JND_Interval_Duration;
	JND_Reference_Intervals_Positions = std::vector<int>(p.JND_Reference_Intervals_Positions);
	JND_Interval_To_Reference = std::vector<int>(p.JND_Interval_To_Reference);
	JND_Calculated_Intervals_Positions = std::vector<int>(p.JND_Calculated_Intervals_Positions);
	JND_Interval_Head = p.JND_Interval_Head;
	JND_Interval_Tail = p.JND_Interval_Tail;
	JND_File_Name = p.JND_File_Name;
	JND_Noise_Source = p.JND_Noise_Source;
	JND_WN_Additional_Intervals = std::vector<float>(p.JND_WN_Additional_Intervals);
	Show_JND_Configuration = p.Show_JND_Configuration;
	Calculate_JND_Types = p.Calculate_JND_Types;
	JND_Delta_Alpha_Time_Factor = p.JND_Delta_Alpha_Time_Factor;
	W = std::vector<float>(p.W);
	Normalize_Noise_Energy_To_Given_Interval = p.Normalize_Noise_Energy_To_Given_Interval;
	Remove_Generated_Noise_DC = p.Remove_Generated_Noise_DC;
	Calculate_JND_On_CPU = p.Calculate_JND_On_CPU;
	M_tot = std::vector<float>(p.M_tot);
	Noise_Expected_Value_Accumulating_Boundaries= std::vector<double>(p.Noise_Expected_Value_Accumulating_Boundaries);
	Noise_Sigma_Accumulating_Boundaries =  std::vector<double>(p.Noise_Sigma_Accumulating_Boundaries);
	Normalize_Sigma_Type = p.Normalize_Sigma_Type;
	Normalize_Sigma_Type_Signal = p.Normalize_Sigma_Type_Signal;
	// complex input generation patterns
	testedFrequencies = std::vector<float>(p.testedFrequencies);
	testedNoises = std::vector<float>(p.testedNoises);
	testedPowerLevels = std::vector<float>(p.testedPowerLevels);
	generatedInputProfile = p.generatedInputProfile;
	showGeneratedConfiguration = p.showGeneratedConfiguration;
	// debug flags
	Review_AudioGramCall = p.Review_AudioGramCall;
	Review_Lambda_Memory = p.Review_Lambda_Memory;
	Show_Generated_Input = p.Show_Generated_Input;
	maxdBSPLJND = p.maxdBSPLJND;
	hasJND_Raw_Output = p.hasJND_Raw_Output;
	JND_Raw_File_Name = p.JND_Raw_File_Name;
	Approximated_JND_EPS_Diff = p.Approximated_JND_EPS_Diff;
	Calculate_From_Mean_Rate = p.Calculate_From_Mean_Rate;
	Discard_BM_Velocity_Output = p.Discard_BM_Velocity_Output;
	Discard_Lambdas_Output = p.Discard_Lambdas_Output;
	Show_Device_Data = p.Show_Device_Data;
	Perform_Memory_Test = p.Perform_Memory_Test;
	Complex_Profile_Noise_Level_Fix_Factor = p.Complex_Profile_Noise_Level_Fix_Factor;
	Complex_Profile_Noise_Level_Fix_Addition = p.Complex_Profile_Noise_Level_Fix_Addition;
	Remove_Transient_Tail = p.Remove_Transient_Tail;
	Cuda_Outern_Loops = p.Cuda_Outern_Loops;
	Run_Fast_BM_Calculation = p.Run_Fast_BM_Calculation;
	IS_MEX = p.IS_MEX;
	Verbose_BM_Vectors = std::vector<std::string>(p.Verbose_BM_Vectors);
	Hearing_AID_FIR_Transfer_Function = std::vector<float>(p.Hearing_AID_FIR_Transfer_Function);
	Hearing_AID_IIR_Transfer_Function = std::vector<float>(p.Hearing_AID_IIR_Transfer_Function);
	AC_Filter_Vector = std::vector<double>(p.AC_Filter_Vector);
	Convergence_Tests = p.Convergence_Tests;
    return *this;    // Return ref for multiple assignment
}//end operator=

bool CParams::notEmptyString(char* str) {
	return str[0]!=0&&str[1]!=0;
}

bool CParams::notEmptyString(const std::string& str) {
	return str!="";
}
/*
int CParams::get_two_tokens(char *lline, char *param, char *val)
{
	int ind=0;

	char ch = lline[0]; 
	while(ch==' ' || ch=='\n' || ch=='\t') 
	{
		ind++;
		ch = lline[ind];		
	}

	int pind=0;
	//printf("ch = %c\n",ch);
	while(ch!=' ' && ch!='\n' && ch!='\t' && ch!='=' && ch!=0)
	{
		param[pind]=ch;
		
	    //printf("param[%d] = ch = %c\n",pind,ch);
		pind++;
		ind++;
		ch = lline[ind];
	}
	param[pind]=0;
	
	while (ch!='=' && ch!=0)
	{
		ind++;
		ch = lline[ind];
	}
	
	if (ch!='=') return 0;

	ind++;
	ch = lline[ind];

	while(ch==' ' || ch=='\n' || ch=='\t') 
	{
		ind++;
		ch = lline[ind];		
	}

	if (ch==0) return 0;

	int string_state=0;
	int string_mask=0;
	int ignore_ch = 0;
	int vind=0;
	while(((ch!=' ' && ch!='\n' && ch!='\t')||(string_mask==1)) && ch!=0)
	{
		ignore_ch = 0;
		if (ch=='"')
		{
			if (string_state==0)
			{
				string_state=1;
				string_mask=1;
				ignore_ch = 1;
			}
			else if (string_state==1)
			{
				string_state=2;
				string_mask=0;
				ignore_ch = 1;
			}
		}
		if (!ignore_ch)
		{
			val[vind]=ch;
			vind++;
		}
		ind++;
		ch = lline[ind];
	}
	val[vind]=0;

	return 1;
}
*/
/*
void CParams::get_line(FILE *fp, char *lline)
{
	int ind = 0;
	int finished = 0;
	while(!feof(fp) && !finished)
	{
		char ch = fgetc(fp);
		if ((ch=='\n')||feof(fp))
		{
			lline[ind]=0;
			finished = 1;
		}
		else
		{
			lline[ind]=ch;
		}
		ind++;
	} 
}

*/
bool CParams::advanceOnNotNumeric(char* argv[], int &position,int argc) {
	// handling all non numeric params on performace test
	while ( position<argc && strspn(argv[position], "0123456789.") != strlen(argv[position]) && strncmp("-", argv[position], sizeof(char)) != 0 ) {
		//printf("non numeric and its %s\n",argv[position]);
		if ( !strcmp(argv[position],"Disable_Lambda") ) {
			disable_lambda = 1;
		}
		position++;
	}
	return (position>=argc||!strncmp("-", argv[position], sizeof(char)));
}
void CParams::parse_arguments_performance(char* argv[], int &position,int argc) {
	//printf("parsing at position %d\n",position);
	position++;
	if (advanceOnNotNumeric(argv,position,argc)) return;
	if (position < argc) sin_freq = parseToScalar<float>(argv[position++]);
	if (advanceOnNotNumeric(argv,position,argc)) return;
	if (position < argc) sin_dB = parseToScalar<float>(argv[position++]);
	if (advanceOnNotNumeric(argv,position,argc)) return;
	sin_amp = CONV_dB_TO_AMP(sin_dB);
	//printf("done parsing at position %d\n",position);	
}


/*
int CParams::calcLastSavedTimeBlock() {
	int baseNodes = calcTimeBlockNodes();
	int totalNodes = baseNodes*Time_Blocks;
	int minimum_necessary_time_nodes = static_cast<int>(Fs*MIN_LAST_SAVED_TIME*__tmax(1.0f, static_cast<float>(Fs)/ RATE_FREQUENCY));
	// totalNodes must be divisible by THREADS_PER_IHC_FILTER_SECTION
	int miniModNodes = THREADS_PER_IHC_FILTER_SECTION - (totalNodes % THREADS_PER_IHC_FILTER_SECTION); // ensure completeion of time nodes
	//miniModNodes = boost::math::lcm(miniModNodes, Time_Blocks);
	while (miniModNodes < minimum_necessary_time_nodes) miniModNodes += THREADS_PER_IHC_FILTER_SECTION; // ensures long enough tail + divisible by THREADS_PER_IHC_FILTER_SECTION
	
	cout << "last saved time block time nodes are: " << miniModNodes << "\n" 
		<< "minimum necessary time nodes " << minimum_necessary_time_nodes<<"\n"
		<< "total nodes " << totalNodes<<"\n"
		<<"base nodes "<< baseNodes<<"\n";
	return miniModNodes;
}
*/
#ifdef CUDA_MEX_PROJECT
void CParams::parseMexFile(const mxArray *input_arrays[], mxArray *output_arrays[]) {
	const mxArray *input_struct = input_arrays[0];
	mxArray *output_struct;
	std::cout << "processing mex strcuture " << std::endl;
	if (vh != NULL) {
		delete vh;
		vh = NULL;
	}
	vh = new MEXHandler(input_struct);
	
	if (vhout != NULL) {
		delete vhout;
		vhout = NULL;

	}
	vhout = new MEXHandler();
	std::vector<std::string> minor_outputs2 = std::vector<std::string>();
	minor_outputs2.push_back("output_struct");
	auto char_matrix = createCharMatrix(minor_outputs2);
	
	output_name = "MEX_TARGET";
	vhout->setPrimaryMajor(output_name);
	output_struct = mxCreateStructMatrix(1, 1, static_cast<int>(char_matrix.size()), char_matrix.data());
	output_arrays[0] = output_struct;
	vhout->setTargetsForWrite(output_name, minor_outputs2, output_struct);
	vhout->writeString("output_struct", "anchor loaded on mex run");
	parse_params_map(false);
}
#endif

void CParams::parse_parameters_file(char *pfname,bool Review_Parse) {
	//FILE *pfp;
	//char in_line[MAX_LONG_INPUT_LENGTH];
	//char in_parameter[MAX_LONG_INPUT_LENGTH];
	//char in_value[MAX_LONG_INPUT_LENGTH];
	if (Review_Parse) {
		printf("Parsing file %s\n",pfname);
	}
	std::string file_name(pfname);
	regex file_type("\\.([a-zA-Z]+)$");
	smatch typefinder;
	if (regex_search(file_name, typefinder, file_type)) {
		const std::string sr(typefinder[1]);
		const std::string utype = transformString(sr,std::toupper<char>);//getFileType(file_name);
		//std::cout << "type of file found : " << utype  << std::endl;
		if (utype.compare("PAR")==0) {
			//std::cout << "processing par file : " << file_name << std::endl;
			vh = new ConfigFileHandler(file_name);
		} 

#ifdef MATLAB_MEX_FILE
		else if (utype.compare("MAT") == 0) {
				std::cout << "processing mat file : " << file_name << std::endl;
				vh = new MEXHandler(file_name);
				vh->processData();
				if (!vh->dataOk()) {
					std::ostringstream oss("");
					oss << "mat file openning failed dut to error code " << vh->getHandlerErrorCode();
					throw std::runtime_error(oss.str());
				}
			} 
#endif
			else {
			std::ostringstream oss("");
			oss << "cannot read config file of format " << utype <<", aborts...";
			throw std::runtime_error(oss.str());
		}
	}
	//ConfigFileHandler(std::string(pfname)); 
	/*
	fopen_s(&pfp, pfname, "r");
	if (!pfp)
	{
		std::cout<< "Can't open parameters file "<<pfname<<std::endl; 
		throw std::runtime_error("Can't open parameters file - 21");
	}
	while (!feof(pfp)) {
		get_line(pfp, in_line);
		//printf("get line %s\n",in_line);
		if (get_two_tokens(in_line, in_parameter, in_value)) {
			paramsMap.insert(pair<std::string, std::string>(std::string(in_parameter), std::string(in_value)));
		}
	}
	fclose(pfp); 
	  */
	//printf("tokens %s = %s\n",in_parameter,in_value);
	
	PrintFormat("Configuration loaded successfully...\n");
	parse_params_map(Review_Parse);
	PrintFormat("Configuration parsed successfully...\n");
	if (Review_Parse) {
		PrintFormat("Configuration file closed...\n");
	}
}

std::string CParams::showJNDConfiguration(const device_jnd_params& config,const int& index) {
	ostringstream oss;
	oss.str("");
	oss.setf(oss.boolalpha);
	oss.setf(std::ios::scientific, std::ios::floatfield);
	oss.precision(3);
	oss << "JND Configuration #" << index << "\n";
	oss << "input["<<index<<"].isRefrence = " << config.isReference << "\n";
	oss << "input[" << index << "].referenceIndex = " << config.referenceIndex << "\n";
	oss << "input[" << index << "].calculateIndexPosition = " << config.calculateIndexPosition << "\n";
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(1);
	oss << "input[" << index << "].dBSPLSignal = " << config.dBSPLSignal << "\n";
	oss << "input[" << index << "].dBSPLNoise = " << config.dBSPLNoise << "\n";
	oss.setf(std::ios::scientific, std::ios::floatfield);
	oss.precision(3);
	oss << "input[" << index << "].dA = " << config.dA << " Dyn\n";
	oss << "input[" << index << "].recipdA = " << config.recipdA << "\n";
	oss << "input[" << index << "].Wn = " << config.Wn << "\n";
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(0);
	oss << "input[" << index << "].frequency = " << config.frequency << "HZ\n";
	return oss.str();
}

std::string CParams::getNoiseName() {
	ostringstream oss;
	oss.str("");
	if (JND_Noise_Source > 0) {
		if (Noise_Name.empty())	oss << "CustomNoise";
		else oss << Noise_Name;
	} else {
		oss << "White Noise";
	}
	return oss.str();
}
std::string CParams::getSignalName() {
	ostringstream oss;
	oss.str("");
	if (JND_Signal_Source > 0) {
		if (Signal_Name.empty())	oss << "CustomSignal";
		else oss << Signal_Name;
	} else {
		oss << "";
	}
	return oss.str();
}
void CParams::updateInputProfile() {
	int tflevel = static_cast<int>(JNDInputTypes());
	int tplevel = static_cast<int>(testedPowerLevels.size());
	int tnlevel = static_cast<int>(testedNoises.size());
	if (tnlevel == 0) { throw std::runtime_error("No noises are present, abort..."); }
	int addFrequencyInterval = Generate_One_Ref_Per_Noise_Level ? 0 : 1;
	int addNoiseLevelInterval = 1 - addFrequencyInterval;
	if (!Calculate_JND) {
		addFrequencyInterval = 0;
		addNoiseLevelInterval = 0;
	}
	complexProfiles.clear();
	if (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS) {
		complexProfiles.reserve(numberOfApproximatedJNDsPerMethod());
		for (int load_run = 0; load_run < numberOfApproximatedJNDsPerMethod(); load_run++) {
			complexProfiles.push_back(ComplexJNDProfile());
		}
	}
	int numOfProfiles = ((static_cast<int>(JNDInputTypes()) + addFrequencyInterval)*static_cast<int>(testedPowerLevels.size()) + addNoiseLevelInterval)*static_cast<int>(testedNoises.size());
	//std::cout << "numberof profiles to be outputted " << numOfProfiles << ", complexProfiles.size() = " << complexProfiles.size() << std::endl;
	if (showGeneratedConfiguration) {
		PrintFormat("Num Signals = %d\n", tflevel);
		PrintFormat("Num Power Levels = %d\n", tplevel);
		PrintFormat("Num Noise Levels = %d\n", tnlevel);
		PrintFormat("Num Of Profiles = %d\n", numOfProfiles);
		if (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS) {
			PrintFormat("Num Of Aggragated Profiles Per Method = %d\n", numberOfApproximatedJNDsPerMethod());
			PrintFormat("Num Of Aggragated Profiles = %d\n", numberOfApproximatedJNDs());
		}
	}
	inputProfile = std::vector<device_jnd_params>(numOfProfiles);
	int calculatedIndex = 0;
	int generalIndex = 0;
	JND_Reference_Intervals_Positions = std::vector<int>(Generate_One_Ref_Per_Noise_Level?tnlevel:tplevel*tnlevel, 0);
	for (int nlevel = 0; nlevel < tnlevel; nlevel++) {
		double noisePower = SPL2PA(testedNoises[nlevel]);
		for (int plevel = 0; plevel < tplevel; plevel++) {
			
			for (int flevel = 0; flevel < tflevel; flevel++) {
				// find correct base signal level according to noise 
				size_t complex_profile_id = tflevel*nlevel + flevel;
				float basesignaldBSPL = testedPowerLevels[plevel];
				if (!Complex_Profile_Power_Level_Divisor.empty() && Complex_Profile_Power_Level_Divisor.size() > complex_profile_id && Complex_Profile_Power_Level_Divisor[complex_profile_id] != 0 && testedNoises[nlevel] != MIN_INF_POWER_LEVEL && testedNoises[nlevel] > 0) {
					basesignaldBSPL = testedPowerLevels[0] + (testedPowerLevels[plevel] - testedPowerLevels[0])*static_cast<float>(Complex_Profile_Power_Level_Divisor[complex_profile_id]) / testedNoises[nlevel];
				}
				if (Complex_Profile_Power_Level_Divisor.size() <= complex_profile_id) {
					PrintFormat("warning: Complex_Profile_Power_Level_Divisor.size(%d) <= complex_profile_id(%d)\n", Complex_Profile_Power_Level_Divisor.size(), complex_profile_id);
				}
				if (testedNoises[nlevel] != MIN_INF_POWER_LEVEL) basesignaldBSPL += static_cast<float>(Complex_Profile_Noise_Level_Fix_Factor[complex_profile_id])*testedNoises[nlevel] + static_cast<float>(Complex_Profile_Noise_Level_Fix_Addition[complex_profile_id]);
				double signalPower = SPL2PA(basesignaldBSPL);


				inputProfile[generalIndex].dBSPLSignal = basesignaldBSPL;
				inputProfile[generalIndex].dBSPLNoise = testedNoises[nlevel] != MIN_INF_POWER_LEVEL ? testedNoises[nlevel] : 0.0f;
				
				inputProfile[generalIndex].isReference = false;
				inputProfile[generalIndex].referenceIndex = -1;
				if (Calculate_JND) inputProfile[generalIndex].referenceIndex = Generate_One_Ref_Per_Noise_Level ? (tplevel*tflevel*(nlevel + 1) + nlevel) : (generalIndex + tflevel - flevel); // basicallyu after the frequencies
				inputProfile[generalIndex].calculateIndexPosition = calculatedIndex;
				inputProfile[generalIndex].dA = signalPower;
				inputProfile[generalIndex].recipdA = signalPower>0?(1.0/signalPower):0;
				inputProfile[generalIndex].Wn = noisePower;
				inputProfile[generalIndex].frequency = JND_Signal_Source?1:testedFrequencies[flevel];
				if (showGeneratedConfiguration) {
					std::cout << showJNDConfiguration(inputProfile[generalIndex], generalIndex);
				}
				if (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS) {
					int cplevel = nlevel*tflevel + flevel;
					if (complexProfiles[cplevel]._frequency == 0.0f) {
						complexProfiles[cplevel].updateBaseValues(JND_Signal_Source ? 1 : testedFrequencies[flevel], testedNoises[nlevel], tplevel,Signal_Name);
					}
					complexProfiles[cplevel].setInterval(calculatedIndex, plevel);
					if (showGeneratedConfiguration&&plevel==tplevel-1) {
						std::cout << complexProfiles[cplevel].show(cplevel);
					}
				}
				calculatedIndex++;
				generalIndex++;
			}
			if (!Generate_One_Ref_Per_Noise_Level && Calculate_JND) {
				JND_Reference_Intervals_Positions[tplevel*nlevel + plevel] = generalIndex;
				inputProfile[generalIndex].dBSPLSignal = 0.0f;
				inputProfile[generalIndex].dBSPLNoise = testedNoises[nlevel] != MIN_INF_POWER_LEVEL ? testedNoises[nlevel] : 0.0f;
				inputProfile[generalIndex].isReference = true;
				inputProfile[generalIndex].referenceIndex = generalIndex; // basicallyu after the frequencies
				inputProfile[generalIndex].calculateIndexPosition = -1;
				inputProfile[generalIndex].dA = 0;
				inputProfile[generalIndex].recipdA = 0;
				inputProfile[generalIndex].Wn = noisePower;
				inputProfile[generalIndex].frequency = 0.0f;
				if (showGeneratedConfiguration) {
					std::cout << showJNDConfiguration(inputProfile[generalIndex], generalIndex);
				}
				generalIndex++;
			}
		}
		if (Generate_One_Ref_Per_Noise_Level && Calculate_JND) {
			JND_Reference_Intervals_Positions[nlevel] = generalIndex; // tplevel*tflevel*(nlevel+1) + nlevel  
			inputProfile[generalIndex].dBSPLSignal = 0.0f;
			inputProfile[generalIndex].dBSPLNoise = testedNoises[nlevel] != MIN_INF_POWER_LEVEL ? testedNoises[nlevel] : 0.0f;
			inputProfile[generalIndex].isReference = true;
			inputProfile[generalIndex].referenceIndex = generalIndex; // basicallyu after the frequencies
			inputProfile[generalIndex].calculateIndexPosition = -1;
			inputProfile[generalIndex].dA = 0;
			inputProfile[generalIndex].recipdA = 0;
			inputProfile[generalIndex].Wn = noisePower;
			inputProfile[generalIndex].frequency = 0.0f;
			if (showGeneratedConfiguration) {
				std::cout << showJNDConfiguration(inputProfile[generalIndex], generalIndex);
			}
			generalIndex++;
		}
	}
	if (generalIndex > static_cast<int>(inputProfile.size())) {
		PrintFormat("FAILURE: inserted #%d number of profiles to %lu sized array\n",generalIndex,inputProfile.size());
		throw std::runtime_error("update input profile failed");
	}
	if (showGeneratedConfiguration && Calculate_JND) {
		PrintFormat("JND_Reference_Intervals_Positions: %s\n", viewVector<int>(JND_Reference_Intervals_Positions).c_str());
	}
	generatedInputProfile = true;
}
// parse loaded map from matlab configuration
void CParams::parse_params_map(bool Review_Parse) {

		 
	if (vh->hasVariable("Sim_Type")) {
		sim_type = vh->getValue<int>("Sim_Type");
	}
	if (vh->hasVariable("Fs")) {
		Fs = vh->getValue<int>("Fs");
		//std::cout << " virtual handler found Fs: " << Fs << std::endl;
		if (Fs < MIN_SAMPLE_RATE) {
			std::ostringstream oss("");
			oss << "Sample Rate cant be below " << MIN_SAMPLE_RATE << "HZ, but its " << Fs << " HZ.";
			
			throw std::runtime_error(oss.str());
		}
	}



	if (vh->hasVariable("Show_Transient")) {
		show_transient = vh->getValue<int>("Show_Transient");
		if (show_transient != 0 && show_transient != 1) {
			show_transient = 1;
		}
	}

	if (vh->hasVariable("BMOHC_Kernel_Configuration")) {
		int prev_config = BMOHC_Kernel_Configuration;
		BMOHC_Kernel_Configuration = vh->getValue<int>("BMOHC_Kernel_Configuration");
		if (prev_config != BMOHC_Kernel_Configuration) {
			PrintFormat("Warning: Cuda running configuration replaced from %s to %s\n", BMOHC_Kernel_Configurations_Names[prev_config].c_str(), BMOHC_Kernel_Configurations_Names[BMOHC_Kernel_Configuration].c_str());
		}
	}
	if (vh->hasVariable("Examine_Configurations")) {
		Examine_Configurations = vh->getValue<int>("Examine_Configurations");
	}
	if (vh->hasVariable("Discard_BM_Velocity_Output")) {
		Discard_BM_Velocity_Output = vh->getValue<int>("Discard_BM_Velocity_Output") == 1;
		if (!Discard_BM_Velocity_Output) Allowed_Outputs |= 1;
	}

	if (vh->hasVariable("Discard_Lambdas_Output")) {
		Discard_Lambdas_Output = vh->getValue<int>("Discard_Lambdas_Output") == 1;
		if (!Discard_Lambdas_Output) Allowed_Outputs |= 14;
	}

	if ( vh->hasVariable("Allowed_Outputs") ) {
		Allowed_Outputs = vh->getValue<unsigned int>("Allowed_Outputs");
	}

	Decouple_Filter = Generating_Input_Profile() || vh->hasVariable("Decouple_Filter") ? vh->getValue<int>("Decouple_Filter") : 0;
	Remove_Transient_Tail = Generating_Input_Profile() ? 1 : 0; // on input profile, tail will be removed
	if (vh->hasVariable("Remove_Transient_Tail")) {
		Remove_Transient_Tail = vh->getValue<int>("Remove_Transient_Tail");
	}
	if (vh->hasVariable("Time_Block_Length")) {
		Time_Block_Length = vh->getValue<float>("Time_Block_Length");
		Time_Block_Length_Microseconds = toMicroSeconds(Time_Block_Length);
		if (Time_Block_Length_Microseconds < TIME_SECTION_SHORT && Decouple_Filter==0) {
			std::ostringstream oss("");
			oss << "duration of time synchronized processed data must be at least " << TIME_SECTION_SHORT << " micro seconds, but its " << Time_Block_Length_Microseconds << " micro seconds.";
			
			throw std::runtime_error(oss.str());
		}
	}
	if (vh->hasVariable("Time_Blocks")) {
		Time_Blocks = vh->getValue<int>("Time_Blocks");
	}
	if (vh->hasVariable("Show_Calculated_Power")) {
		Show_Calculated_Power = vh->getValue<int>("Show_Calculated_Power");
	}
	if (vh->hasVariable("Processed_Interval")) {
		Processed_Interval = vh->getValue<double>("Processed_Interval");
		Time_Blocks = static_cast<int>(round(Processed_Interval / Time_Block_Length));
		if (vh->hasVariable("Time_Blocks")) {
			throw std::runtime_error("Ambiguty on Time_Blocks/Processed_Interval, please set only one. aborts");
		}
	}

	if (Decouple_Filter > 0 && show_transient == 0) {
		stringstream oss("");
		oss << "Transient must be shown when Decouple_Filter > 0 (=" << Decouple_Filter << ")";
		throw std::runtime_error(oss.str());
	} else if (Decouple_Filter > Time_Blocks || (Decouple_Filter > 0 && Time_Blocks%Decouple_Filter > 0)) {
		stringstream oss("");
		oss << "Decouple_Filter(=" << Decouple_Filter << ") cannot be larger than Time_Blocks(=" << Time_Blocks << ") and has to divide them without remainder(=" << (Time_Blocks%Decouple_Filter) << ")";
		throw std::runtime_error(oss.str());
	}
	if (vh->hasVariable("Overlap_Time")) {
		setOverlapTime(vh->getValue<double>("Overlap_Time"));
		if (overlapTimeMicroSec < TIME_OFFSET  && Decouple_Filter==0) {
			stringstream oss("");
			oss << "duration of time synchronized processed data must be at least " << TIME_OFFSET << " micro seconds, but its " << Time_Block_Length_Microseconds << " micro seconds.";
			throw std::runtime_error(oss.str());
			
		}
	}

	
	//if (Decouple_Filter == 1)  setOverlapTime(0.0);
	
	if (vh->hasVariable("Continues")) {
		continues = vh->getValue<int>("Continues");
	}
	if (vh->hasVariable("Convergence_Tests")) {
		Convergence_Tests = vh->getValue<unsigned int>("Convergence_Tests");
	}
	if (vh->hasVariable("JND_USE_Spont_Base_Reference_Lambda") && vh->getValue<int>("Max_dB_SPL_JND") == 0) {
		JND_USE_Spont_Base_Reference_Lambda = 0;
	}

	if (vh->hasVariable("Max_dB_SPL_JND")) {
		maxdBSPLJND = vh->getValue<float>("Max_dB_SPL_JND");
	}
	if (vh->hasVariable("Complex_JND_Calculation_Types")) {
		Complex_JND_Calculation_Types = vh->getValue<unsigned int>("Complex_JND_Calculation_Types")&ACTIVE_JND_FLAGS;
		if (Complex_JND_Calculation_Types == 0) Complex_JND_Calculation_Types = 1; // this cant be zero at least one method must be chosen
	}


	if (vh->hasVariable("M_tot")) {
		std::vector<float> sv = vh->getValue<std::vector<float> >("M_tot");
		if (!sv.empty()) {
			M_tot = sv;
		}
	}
	M_tot = expandVectorToSize(M_tot, static_cast<std::size_t>(SECTIONS));

	if (vh->hasVariable("Sin_Freq")) {
		sin_freq = vh->getValue<float>("Sin_Freq");
	}

	if (vh->hasVariable("Scale_BM_Velocity_For_Lambda_Calculation")) {
		scaleBMVelocityForLambdaCalculation = vh->getValue<float>("Scale_BM_Velocity_For_Lambda_Calculation");
	}

	if (vh->hasVariable("Calculate_JND")) {
		Calculate_JND = vh->getValue<int>("Calculate_JND") == 1;
	}

	if (vh->hasVariable("Disable_Lambda")) {
		disable_lambda = vh->getValue<int>("Disable_Lambda") == 1 && !Calculate_JND; // calculate JND override disable lambda
	}
	if (vh->hasVariable("Force_IHC_on_CPU")) {
		run_ihc_on_cpu = vh->getValue<int>("Force_IHC_on_CPU") == 1;
	}

	
	if (vh->hasVariable("Show_Filter")) {
		show_filter = vh->getValue<int>("Show_Filter") == 1;
	}
	
	if (vh->hasVariable("Review_Lambda_Memory")) {
		Review_Lambda_Memory = vh->getValue<int>("Review_Lambda_Memory") == 1;
	}
	if (vh->hasVariable("Perform_Memory_Test")) {
		Perform_Memory_Test = vh->getValue<int>("Perform_Memory_Test") == 1;
	}
	if (vh->hasVariable("Review_AudioGramCall")) {
		Review_AudioGramCall = vh->getValue<int>("Review_AudioGramCall") == 1;
	}
	if (vh->hasVariable("Run_Fast_BM_Calculation")) {
		Run_Fast_BM_Calculation = vh->getValue<int>("Run_Fast_BM_Calculation");
	}

	
	if (vh->hasVariable("Disable_Advanced_Memory_Handling")) {
		disable_advanced_memory_handling = vh->getValue<int>("Disable_Advanced_Memory_Handling") == 1;
	}
	if (vh->hasVariable("Show_JND_Configuration")) {
		Show_JND_Configuration = vh->getValue<int>("Show_JND_Configuration") == 1;
	}
	if (vh->hasVariable("Offset")) {
		offset = vh->getValue<float>("Offset");
	}

	// maximum time step for algorithm
	if (vh->hasVariable("MAX_TIME_STEP")) {
		cuda_max_time_step = std::powf(10.0f,-0.1f*vh->getValue<float>("MAX_TIME_STEP"));
	}


	// minimum time step for algorithm
	if (vh->hasVariable("MIN_TIME_STEP")) {
		cuda_min_time_step = std::powf(10.0f, -0.1f*vh->getValue<float>("MIN_TIME_STEP"));
	}



	if (vh->hasVariable("SPLRef")) {
		SPLRefVal = vh->getValue<double>("SPLRef");
	}


	if (vh->hasVariable("MAX_M1_SP_ERROR")) {
		Max_M1_SP_Error_Parameter = powf(10.0f, vh->getValue<float>("MAX_M1_SP_ERROR"));
	}


	if (vh->hasVariable("MAX_TOLERANCE")) {
		Max_Tolerance_Parameter = powf(10.0f, vh->getValue<float>("MAX_TOLERANCE"));
	}

	if (vh->hasVariable("RELATIVE_ERRORS")) {
		Relative_Error_Parameters = vh->getValue<int>("RELATIVE_ERRORS");
	}

	if (vh->hasVariable("Decouple_Unified_IHC_Factor")) {
		Decouple_Unified_IHC_Factor = vh->getValue<int>("Decouple_Unified_IHC_Factor");
	}


	if (vh->hasVariable("Show_Run_Time")) {
		Show_Run_Time = vh->getValue<int>("Show_Run_Time");
	}

	if (vh->hasVariable("Show_CPU_Run_Time")) {
		Show_CPU_Run_Time = vh->getValue<int>("Show_CPU_Run_Time");
	}

	if (vh->hasVariable("Tolerance_Fix_Factor")) {
		Tolerance_Fix_Factor = vh->getValue<float>("Tolerance_Fix_Factor");
	}

	if (vh->hasVariable("M1_SP_Fix_Factor")) {
		M1_SP_Fix_Factor = vh->getValue<float>("M1_SP_Fix_Factor");
	}

	if (vh->hasVariable("Show_Device_Data")) {
		Show_Device_Data = vh->getValue<int>("Show_Device_Data");
	}
	if (vh->hasVariable("Lambda_SAT")) {
		Lambda_SAT = vh->getValue<float>("Lambda_SAT");
	}
	if (vh->hasVariable("eta_AC")) {
		eta_AC = vh->getValue<float>("eta_AC");
	}
	if (vh->hasVariable("eta_DC")) {
		eta_DC = vh->getValue<float>("eta_DC");
	}
	if (vh->hasVariable("Tdelta")) {
		Tdelta = vh->getValue<float>("Tdelta");
	}

	if (vh->hasVariable("JACOBBY_Loops_Fast")) {
		JACOBBY_Loops_Fast = vh->getValue<int>("JACOBBY_Loops_Fast");
	}

	if (vh->hasVariable("JACOBBY_Loops_Slow")) {
		JACOBBY_Loops_Slow = vh->getValue<int>("JACOBBY_Loops_Slow");
	}
	if (vh->hasVariable("Cuda_Outern_Loops")) {
		Cuda_Outern_Loops = vh->getValue<int>("Cuda_Outern_Loops");
	}
	
	if (vh->hasVariable("Load_Previous")) {
		loadPrevData = vh->getValue<int>("Load_Previous");
	}
	
	
	//
	if (vh->hasVariable("Verbose_BM_Vectors")) {
		Verbose_BM_Vectors = splitToVector(vh->getValue<std::string>("Verbose_BM_Vectors"), "([a-zA-Z_]+)");
	}

	if (vh->hasVariable("Load_Previous_File") && notEmptyString(vh->getValue<std::string>("Load_Previous_File"))) {
		strcpy_s(backup_file_name, vh->getValue<std::string>("Load_Previous_File").c_str());
	}
	if (vh->hasVariable("Num_Frequencies")) {
		num_frequencies = vh->getValue<int>("Num_Frequencies");
		frequncies_analysis = 1;
	}
	if (vh->hasVariable("Power_Frequencies")) {

		//std::vector<std::string> strs;
		//boost::split(strs, vh->getValue<std::string>("Power_Frequencies"), boost::is_any_of("\t "));
		std::vector<float> vf = vh->getValue<std::vector<float> >("Power_Frequencies");
		for (int i = 0; i < static_cast<int>(vf.size()); i++) {
			frequncies[i] = vf[i];
		}
		/*
		char *next_token;
		const char *delim = " ";
		char *token = strtok_s(in_value, delim, &next_token);
		while(token && cou < num_frequencies ) {
		frequncies[cou] = parseToScalar<float>(token);
		if (Review_Parse) {
		printf("frequncies[%d] = %.0f\n", cou, frequncies[cou]);
		}
		token = strtok_s(NULL, delim, &next_token);
		cou++;
		}
		*/
	}
	if (vh->hasVariable("Filter_Mode")) {
		ac_filter_mode = vh->getValue<int>("Filter_Mode");
	}


	if (vh->hasVariable("AC_Filter_Vector")) {
		AC_Filter_Vector = vh->getValue<std::vector<double> >("AC_Filter_Vector");
		ac_filter_mode = 2;
	}

	if (vh->hasVariable("Mex_Debug_In_Output")) {
		Mex_Debug_In_Output = vh->getValue<int>("Mex_Debug_In_Output");
	}
	if (vh->hasVariable("Function_Filter") && ac_filter_mode == 1 && !disable_lambda) {

		std::string in_string = vh->getValue<std::string>("Function_Filter");

		std::smatch sm;
		std::regex digit_test("(-?[[:digit:]]+)(\\.(([[:digit:]]+)?))?");
		std::regex params_searcher("([a-zA-Z0-9_]+)=([a-zA-Z0-9_\\.]+)");
		if (Review_Parse) {

			std::cout << "search params...\n";
		}
		while (std::regex_search(in_string, sm, params_searcher)) {
			std::string origFieldName(sm[1]);
			std::string fieldName(origFieldName);
			std::string fieldValue(sm[2]);
			std::string fieldValueRaw(sm[2]);

			if (Review_Parse) {
				std::cout << "found field " << origFieldName << "=" << fieldValueRaw << "\n";
			}
			std::transform(fieldName.begin(), fieldName.end(), fieldName.begin(), ::tolower);
			std::transform(fieldValue.begin(), fieldValue.end(), fieldValue.begin(), ::tolower);
			filtersMap.insert(make_pair(fieldName, fieldValue));
			filtersMapRaw.insert(make_pair(origFieldName, fieldValueRaw));
			filtersKeysStat.insert(make_pair(fieldName, origFieldName));
			// field name type is the only one thats not number otherwise its error
			in_string = sm.suffix().str();
		}

		if (filtersMap.count("type")) {
			filterName.assign(filtersMap["type"]);
		}

		if (filtersMap.count("order")) {
			if (filtersMap["order"] == "minimum") {
				FilterOrder = -1;
			} else {
				FilterOrder = atoi(filtersMap["order"].c_str());
			}

		}
		if (filtersMap.count("mode")) {
			mode.assign(filtersMap["mode"]);
			if (mode.find("constrained") != std::string::npos) deltasFound = true;
			if (mode.find("stopband") != std::string::npos) butterType = STOPBAND;
		}
		if (filtersMap.count("fc") && (filtersMap.count("fstop") || filtersMap.count("fpass"))) {
			std::ostringstream oss("");
			oss << "Filter function '" << vh->getValue<std::string>("Function_Filter") << "' cannot contains both Fc and Fpass/Fstop, aborts...";
			throw std::runtime_error(oss.str());
		}
		if (filtersMap.count("fc")) Fc = parseToScalar<float>(filtersMap["fc"]);
		if (filtersMap.count("fpass")) Fpass = parseToScalar<float>(filtersMap["fpass"]);
		if (filtersMap.count("fstop")) Fstop = parseToScalar<float>(filtersMap["fstop"]);
		if (filtersMap.count("astop")) {
			Astop = parseToScalar<float>(filtersMap["astop"]);
			deltasFound = true;
		}
		if (filtersMap.count("apass")) {
			Apass = parseToScalar<float>(filtersMap["apass"]);
			deltasFound = true;
		}
		if (filtersMap.count("wstop")) Wstop = parseToScalar<float>(filtersMap["wstop"]);
		if (filtersMap.count("wpass")) Wpass = parseToScalar<float>(filtersMap["wpass"]);
		if (filtersMap.count("slope")) Slope = parseToScalar<float>(filtersMap["slope"]);
		if (filtersMap.count("view")) show_filter = atoi(filtersMap["view"].c_str()) == 1;
		if (filtersMap.count("minorder")) minOrder = atoi(filtersMap["minorder"].c_str()) == 1;
		if (filtersMap.count("order")) FilterOrder = atoi(filtersMap["order"].c_str());

		// fir filter window verfication of parameters
		if (filterName == "window") {
			if (filtersMap.count("windowtype") == 0) {
				throw std::runtime_error("FIR filter with Window must have window type");
			}
			if (minOrder) {
				std::cout << "MinOrder is meaningless in window type filter since Fc will need specific order and Fpass,Fstop will automatically calculate Minimum order";
			}
			if (filtersMap.count("fc") && filtersMap.count("order") == 0) {
				// minimum order is meaningless since now transition width is given
				throw std::runtime_error("FIR filter with Fc must get specific Order since no transition width is available");
			}
			if (filtersMap.count("fc") == 0 && (filtersMap.count("fpass") == 0 || filtersMap.count("fstop") == 0)) {
				throw std::runtime_error("FIR filter without Fc must have Fpass,Fstop");
			}
		}


		/**
		* this needs rewrite
		else {
		//if ( !boost::regex_match(fieldValue, digit_test)) {
		cout << "field " << origFieldName << " is invalid, need to be number but is '" << fieldValue << "'\n";
		throw std::runtime_error("field from params file is invalid aborts - 1");
		}
		float val = parseToScalar<float>(fieldValue);
		if (filtersMap.count("fc")) Fc = val;
		else if (filtersMap.count("fstop")) Fstop = val;
		else if (filtersMap.count("fpass")) Fpass = val;
		else if (filtersMap.count("astop"))  {
		Astop = val;
		deltasFound = true;
		}
		else if (filtersMap.count("apass")) {
		Apass = val;
		deltasFound = true;
		}
		else if (filtersMap.count("wstop")) Wstop = val;
		else if (filtersMap.count("wpass")) Wpass = val;
		else if (filtersMap.count("slope")) Slope = val;
		else if (filtersMap.count("view")) show_filter = static_cast<int>(val)==1;
		else if (filtersMap.count("minorder")) minOrder = static_cast<int>(val) == 1;
		else if (filtersMap.count("order")) FilterOrder = static_cast<int>(val);
		}
		*/


	}


	if (vh->hasVariable("Sin_dB")) {
		sin_dB = vh->getValue<float>("Sin_dB");
		sin_amp = CONV_dB_TO_AMP(sin_dB);
		if (Review_Parse) {
			printf("in value = %s -> %f -> %f\n", vh->getValue<std::string>("Sin_dB").c_str(), sin_dB, sin_amp);
		}
	}

	if (vh->hasVariable("Input_Signal")) {
		Input_Signal = vh->getValue<std::vector<double> >("Input_Signal");
	}
	if (vh->hasVariable("Input_Noise")) {
		Input_Noise = vh->getValue<std::vector<double> >("Input_Noise");
	}

	if (vh->hasVariable("Input_File") && notEmptyString(vh->getValue<std::string>("Input_File"))) {
		strcpy_s(in_file_name, vh->getValue<std::string>("Input_File").c_str());
		in_file_flag = 1;
	}
	if (vh->hasVariable("Input_Noise_File") && notEmptyString(vh->getValue<std::string>("Input_Noise_File"))) {
		strcpy_s(in_noise_file_name, vh->getValue<std::string>("Input_Noise_File").c_str());
		input_noise_file_flag = 1;
	}

	if (vh->hasVariable("Signal_Name") && notEmptyString(vh->getValue<std::string>("Signal_Name"))) {
		Signal_Name = vh->getValue<std::string>("Signal_Name");
	}

	if (vh->hasVariable("Noise_Name") && notEmptyString(vh->getValue<std::string>("Noise_Name"))) {
		Noise_Name = vh->getValue<std::string>("Noise_Name");
	}

	if (vh->hasVariable("Run_Stage_File_name") && notEmptyString(vh->getValue<std::string>("Run_Stage_File_name"))) {
		Run_Stage_File_name = vh->getValue<std::string>("Run_Stage_File_name");
	}
	if (vh->hasVariable("Run_Stage_Calculation")) {
		Run_Stage_Calculation = vh->getValue<int>("Run_Stage_Calculation");
	}
	if (Run_Stage_Unverfied()) {
		std::ostringstream oss("");
		oss << "Chosen Run stage calculation (=" << Run_Stage_Calculation << ") need data or file";
		std::string errmsg = oss.str();
		throw std::runtime_error(errmsg);
	}
	if (vh->hasVariable("Type_TEST_Output") && notEmptyString(vh->getValue<std::string>("Type_TEST_Output"))) {
		Type_TEST_Output = vh->getValue<std::string>("Type_TEST_Output");
		//std::cout << "Type_TEST_Output == " << Type_TEST_Output << std::endl;
	}
	if (vh->hasVariable("Gamma_File") && notEmptyString(vh->getValue<std::string>("Gamma_File"))) {
		strcpy_s(gamma_file_name, vh->getValue<std::string>("Gamma_File").c_str());
		gamma_file_flag = 1;
	}
	if (vh->hasVariable("Nerves_File") && notEmptyString(vh->getValue<std::string>("Nerves_File"))) {
		strcpy_s(nerve_file_name, vh->getValue<std::string>("Nerves_File").c_str());
		nerve_file_flag = 1;
	}
	if (vh->hasVariable("OHC_Mode")) {
		ohc_mode = vh->getValue<int>("OHC_Mode");
		//std::cout << "OHC_Mode = " << ohc_mode << std::endl;
		//ohc_mode = vh->getValue<int>("OHC_Mode");
	}
	if (vh->hasVariable("IHC_Mode")) {
		ihc_mode = vh->getValue<int>("IHC_Mode");
	}
	if (vh->hasVariable("OHC_Vector")) {
		ohc_vector = expandVectorToSize(vh->getValue<std::vector<float> >("OHC_Vector"),SECTIONS);
		if (Examine_Configurations_Flags(0)) {
			PrintFormat("ohc vector: %s \n",viewVector(ohc_vector, 16).c_str());
		}
	}
	if (vh->hasVariable("IHC_Vector")) {
		ihc_vector = expandVectorToSize(vh->getValue<std::vector<float> >("IHC_Vector"), SECTIONS);
		if (Examine_Configurations_Flags(1)) {
			PrintFormat("ihc vector: %s \n", viewVector(ihc_vector, 16).c_str());
		}
	}
	if (vh->hasVariable("JND_Include_Legend")) {
		//JND_Include_Legend = vh->getValue<unsigned int>("JND_Include_Legend");
		JND_Include_Legend = vh->getValue<unsigned int>("JND_Include_Legend");
	}
	unsigned int output_found = (vh->hasVariable("Output_Result") && notEmptyString(vh->getValue<std::string>("Output_Result")));
	if (IS_MEX>0 || output_found ) {
		if (IS_MEX==0) {
			//PrintFormat("Out file name loaded\n");
			strcpy_s(output_file_name, vh->getValue<std::string>("Output_Result").c_str());
			output_name = std::string(output_file_name);
			//PrintFormat("Out file name %s loaded\n",output_name.c_str());
			std::string output_type = getFileType(output_name);
			//PrintFormat("Out file type %s loaded\n", output_type.c_str());
			//std::cout << "output result type == '"<<output_type << "'" << endl;
			if (output_type.compare("BIN") == 0) {
				//PrintFormat("Creating config handler\n");
				vhout = new ConfigFileHandler();
				//PrintFormat("Created config handler\n");
			} 
#ifdef MATLAB_MEX_FILE			
			else {
				vhout = new MEXHandler();
			}
#endif
			vhout->setPrimaryMajor(output_name);
		} else {
			
		}
		std::vector<std::string> minor_outputs_names = std::vector<std::string>();
		minor_outputs_names.push_back("output_results");
		minor_outputs_names.push_back("lambda_high");
		minor_outputs_names.push_back("lambda_medium");
		minor_outputs_names.push_back("lambda_low");
		std::vector<std::string> minor_outputs = std::vector<std::string>();
		for (int oi = 0; oi < static_cast<int>(minor_outputs_names.size()); oi++) {
			if (Allowed_Output_Flags(oi)) minor_outputs.push_back(minor_outputs_names[oi]);
		}
		//PrintFormat("allocating output vector\n");
		if (vhout->Is_Matlab_Formatted() == 0) {
			//std::cout << "targets output buffer" << std::endl;
			size_t buffer_allocated_nodes = __tmin(static_cast<int>(Fs*(((vh->hasVariable("Processed_Interval") ? static_cast<float>(Processed_Interval) : duration) + 1.0f)*SECTIONS*(disable_lambda ? 1 : 4))), MAX_OUTPUT_BUFFER_LENGTH);
			PrintFormat("allocating output vector in progress %d\n", buffer_allocated_nodes);
			vhout->setTargetsForWrite(output_name, minor_outputs, new OutputBuffer<float>(output_file_name, buffer_allocated_nodes, false));
			PrintFormat("allocated output vector\n");
		} else {
			//std::cout << "targets matlab buffer" << std::endl;
			std::vector<std::string> minor_outputs2 = std::vector<std::string>(minor_outputs);
#ifdef MATLAB_MEX_FILE	
			if (IS_MEX == 0) {
				minor_outputs2.push_back("output_struct");
				auto char_matrix = createCharMatrix(minor_outputs2);
				vhout->writeString("output_struct", "anchor loaded");
				vhout->setTargetsForWrite(output_name, minor_outputs, mxCreateStructMatrix(1, 1, static_cast<int>(char_matrix.size()), char_matrix.data()));
			}
#endif
			vhout->writeVectorString("image_data", minor_outputs, 1);
			/**
			* pre allocation unnecessary due to mexplus
			if (!minor_outputs.empty()) {
				for (auto minor : minor_outputs) {
					std::cout << "minor found: " << minor << std::endl;
					vhout->preAllocateMinor(minor, output_name, static_cast<size_t>(SECTIONS), totalTimeNodes() - (overlapnodesFixReduce(0.0) / SECTIONS));
				}
			}
			*/
		}
	}
	if (vh->hasVariable("Lambda_High") && notEmptyString(vh->getValue<std::string>("Lambda_High"))) {
		strcpy_s(lambda_high_file_name, vh->getValue<std::string>("Lambda_High").c_str());
	}
	if (vh->hasVariable("Lambda_Medium") && notEmptyString(vh->getValue<std::string>("Lambda_Medium"))) {
		strcpy_s(lambda_medium_file_name, vh->getValue<std::string>("Lambda_Medium").c_str());
	}
	if (vh->hasVariable("Lambda_Low") && notEmptyString(vh->getValue<std::string>("Lambda_Low"))) {
		strcpy_s(lambda_low_file_name, vh->getValue<std::string>("Lambda_Low").c_str());
	}
	if (vh->hasVariable("AC_Filter_File") && notEmptyString(vh->getValue<std::string>("AC_Filter_File"))) {
		strcpy_s(ac_filter_file_name, vh->getValue<std::string>("AC_Filter_File").c_str());
	}
	bool duration_flagged = false;
	if (vh->hasVariable("Duration")) {
		duration_flagged = true;
		duration = vh->getValue<float>("Duration");
		if (Review_Parse) {
			printf("Duration = %f !!!!\n", duration);
		}
	}


	if (vh->hasVariable("Aihc")) {
		std::vector<float> aihc_vs = vh->getValue<std::vector<float> >("Aihc");
		if (aihc_vs.size() > 0) {
			Aihc.resize(aihc_vs.size());
			//Aihc = std::vector<float>(aihc_vs);
			std::copy(aihc_vs.begin(), aihc_vs.end(), Aihc.begin());
			printf("Aihc[0]=%.2f\n", Aihc[0]);
		}
	}
	if (Aihc.size() < SECTIONS*LAMBDA_COUNT) {
		Aihc = expandVectorToSize(Aihc, SECTIONS*LAMBDA_COUNT);
	}

	if (vh->hasVariable("spontRate")) {
		spontRate = vh->getValue<std::vector<float> >("spontRate");
	}


	//if (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS) Calculate_JND = true;
	if (Generating_Input_Profile() || Calculate_JND) {
		if (duration_flagged&&Generating_Input_Profile()) {
			// duration will not be flagged 
			stringstream ossfailed("");
			ossfailed << "Duration is calculated from other parameters and shouldnt be set value is " << duration;
			throw std::runtime_error(ossfailed.str());
		}
		if (vh->hasVariable("Hearing_AID_FIR_Transfer_Function")) {
			std::vector<float> sv = vh->getValue<std::vector<float> >("Hearing_AID_FIR_Transfer_Function");
			if (!sv.empty()) {
				Hearing_AID_FIR_Transfer_Function = sv;
			}
		}

		if (vh->hasVariable("Hearing_AID_IIR_Transfer_Function")) {
			std::vector<float> sv = vh->getValue<std::vector<float> >("Hearing_AID_IIR_Transfer_Function");
			if (!sv.empty()) {
				Hearing_AID_IIR_Transfer_Function = -1.0f * sv;
				Hearing_AID_IIR_Transfer_Function[0] = 1.0f;
			}
		}
		if (vh->hasVariable("JND_Interval_Duration")) {
			JND_Interval_Duration = vh->getValue<float>("JND_Interval_Duration");
		}
		if (vh->hasVariable("JND_Interval_Head")) {
			JND_Interval_Head = vh->getValue<float>("JND_Interval_Head");
		}
		//"Calculate_JND_On_CPU"

		if (vh->hasVariable("Calculate_JND_On_CPU")) {
			Calculate_JND_On_CPU = vh->getValue<int>("Calculate_JND_On_CPU") == 1;
		}

		if (vh->hasVariable("JND_Interval_Tail")) {
			JND_Interval_Tail = vh->getValue<float>("JND_Interval_Tail");
		}

		if (vh->hasVariable("JND_Delta_Alpha_Time_Factor")) {
			JND_Delta_Alpha_Time_Factor = vh->getValue<double>("JND_Delta_Alpha_Time_Factor");
		}

		if (vh->hasVariable("JND_Ignore_Tail_Output")) {
			JND_Ignore_Tail_Output = vh->getValue<float>("JND_Ignore_Tail_Output");
		}

		if (vh->hasVariable("JND_Noise_Source")) {
			JND_Noise_Source = vh->getValue<int>("JND_Noise_Source");
		}
		if (vh->hasVariable("JND_Signal_Source")) {
			JND_Signal_Source = vh->getValue<int>("JND_Signal_Source");
		}
	}

	
	if (Generating_Input_Profile()) {
		if (vh->hasVariable("Noise_Sigma_Accumulating_Boundaries")) {
			Noise_Sigma_Accumulating_Boundaries = vh->getValue<std::vector<double> >("Noise_Sigma_Accumulating_Boundaries");
		}
		if (vh->hasVariable("Normalize_Sigma_Type")) {
			Normalize_Sigma_Type = vh->getValue<int>("Normalize_Sigma_Type");
		}
		if (vh->hasVariable("Normalize_Sigma_Type_Signal")) {
			Normalize_Sigma_Type_Signal = vh->getValue<int>("Normalize_Sigma_Type_Signal");
		}
		if (vh->hasVariable("Filter_Noise_Flag")) {
			Filter_Noise_Flag = vh->getValue<int>("Filter_Noise_Flag");
			if (Filter_Noise_Flag > 0) {
				if (vh->hasVariable("Filter_Noise_File") == 0) {
					throw std::runtime_error("Noise marked for filtering please provide file");
				}
				Filter_Noise_File = vh->getValue<std::string>("Filter_Noise_File");
			}
		}
		if (vh->hasVariable("Noise_Expected_Value_Accumulating_Boundaries")) {
			Noise_Expected_Value_Accumulating_Boundaries = vh->getValue<std::vector<double> >("Noise_Expected_Value_Accumulating_Boundaries");
		}

		if (vh->hasVariable("testedFrequencies")) {
			testedFrequencies = vh->getValue<std::vector<float> >("testedFrequencies");
		}
		if (vh->hasVariable("testedNoises")) {
			testedNoises = vh->getValue<std::vector<float> >("testedNoises");
		}
		if (vh->hasVariable("testedPowerLevels")) {
			testedPowerLevels = vh->getValue<std::vector<float> >("testedPowerLevels");
		}
		if (testedPowerLevels.size() > 1) {
			Approximated_JND_EPS_Diff = (testedPowerLevels[1] - testedPowerLevels[0]) / 2;
		}



		if (vh->hasVariable("Complex_Profile_Noise_Level_Fix_Factor")) {
			std::vector<double> tdv = vh->getValue<std::vector<double> >("Complex_Profile_Noise_Level_Fix_Factor");
			if (tdv.size() > 0) {
				Complex_Profile_Noise_Level_Fix_Factor = tdv;
			}
			
		}
		Complex_Profile_Noise_Level_Fix_Factor = replicateAndxpandVector(Complex_Profile_Noise_Level_Fix_Factor, testedNoises.size(), targetJNDComplexProfiles());

		if (vh->hasVariable("Complex_Profile_Noise_Level_Fix_Addition")) {
			std::vector<double> tdv1 = vh->getValue<std::vector<double> >("Complex_Profile_Noise_Level_Fix_Addition");
			if (tdv1.size() > 0) {
				Complex_Profile_Noise_Level_Fix_Addition = tdv1;
			}
			
		}
		Complex_Profile_Noise_Level_Fix_Addition = replicateAndxpandVector(Complex_Profile_Noise_Level_Fix_Addition, testedNoises.size(), targetJNDComplexProfiles());

		if (vh->hasVariable("Complex_Profile_Power_Level_Divisor")) {
			std::vector<double> tdv2 = vh->getValue<std::vector<double> >("Complex_Profile_Power_Level_Divisor");
			if (tdv2.size() > 0) {
				Complex_Profile_Power_Level_Divisor = tdv2;
			}
			
		}
		Complex_Profile_Power_Level_Divisor = replicateAndxpandVector(Complex_Profile_Power_Level_Divisor, testedNoises.size(), targetJNDComplexProfiles());

		

		if (vh->hasVariable("Remove_Generated_Noise_DC")) {
			Remove_Generated_Noise_DC = vh->getValue<double>("Remove_Generated_Noise_DC");
		}

		if (vh->hasVariable("Normalize_Noise_Energy_To_Given_Interval")) {
			Normalize_Noise_Energy_To_Given_Interval = vh->getValue<double>("Normalize_Noise_Energy_To_Given_Interval");
		}

		if (vh->hasVariable("Approximated_JND_EPS_Diff")) {
			Approximated_JND_EPS_Diff = vh->getValue<float>("Approximated_JND_EPS_Diff");
		}
		if (vh->hasVariable("Show_Generated_Configuration")) {
			showGeneratedConfiguration = vh->getValue<int>("Show_Generated_Configuration") == 1;
		}
		if (vh->hasVariable("Show_Generated_Input")) {
			Show_Generated_Input = vh->getValue<int>("Show_Generated_Input");
		}

		if (vh->hasVariable("Generate_One_Ref_Per_Noise_Level")) {
			Generate_One_Ref_Per_Noise_Level = vh->getValue<int>("Generate_One_Ref_Per_Noise_Level") == 1;
		}
		if ( sim_type==SIM_TYPE_JND_COMPLEX_CALCULATIONS && vh->hasVariable("View_Complex_JND_Source_Values")) {
			View_Complex_JND_Source_Values = vh->getValue<int>("View_Complex_JND_Source_Values") == 1;
		}
		duration = static_cast<float>(calculatedDurationGenerated());
		//std::cout << "Duration found: " << duration << "\n";
		CParams::updateInputProfile(); // generated profile for jnd to handle if necessary
	}
	if (Calculate_JND) {

		if (vh->hasVariable("JND_File_Name") && notEmptyString(vh->getValue<std::string>("JND_File_Name"))) {
			JND_File_Name = vh->getValue<std::string>("JND_File_Name");
		}
		else {
			throw std::runtime_error("JND File is missing, cannot output results...");
		}
		if (vh->hasVariable("JND_Raw_File_Name") && notEmptyString(vh->getValue<std::string>("JND_Raw_File_Name"))) {
			JND_Raw_File_Name = vh->getValue<std::string>("JND_Raw_File_Name");
			hasJND_Raw_Output = true;
		}
		if (vh->hasVariable("W")) {
			W = vh->getValue<std::vector<float> >("W");
		}
	}
	if (vh->hasVariable("Calculate_From_Mean_Rate")) {
		Calculate_From_Mean_Rate = vh->getValue<int>("Calculate_From_Mean_Rate") == 1;
	}
	PrintFormat("Calculate_From_Mean_Rate=%d\n", Calculate_From_Mean_Rate);

	if (vh->hasVariable("Calculate_JND_Types")) {
		unsigned int tmp = vh->getValue<unsigned int>("Calculate_JND_Types");
		if (tmp != 0) {
			Calculate_JND_Types = tmp;
		}
	}
	//cout << "note, Calculate_JND_Types: " << Calculate_JND_Types << "\n";
	int number_of_references = 0;
	int number_of_calculated = 0;
	bool tail_intervals = (vh->hasVariable("JND_Tail_Intervals") == 1) && (vh->getValue<int>("JND_Tail_Intervals") >0);
	if (generatedInputProfile) {
		// done on update input profile as base
		//int referencesLength = testedPowerLevels.size()*testedNoises.size();
		//JND_Reference_Intervals_Positions = std::vector<int>(referencesLength, 0);
	} else {
		if (vh->hasVariable("JND_Reference_Intervals_Positions") && notEmptyString(vh->getValue<std::string>("JND_Reference_Intervals_Positions"))) {
			JND_Reference_Intervals_Positions = vh->getValue<std::vector<int> >("JND_Reference_Intervals_Positions");
		} else if (vh->hasVariable("JND_Reference_Rate") || vh->hasVariable("JND_Single_Reference_Rate")) {
			int reference_rate = vh->hasVariable("JND_Reference_Rate") ? vh->getValue<int>("JND_Reference_Rate") : numberOfJNDIntervals();
			int reference_size = numberOfJNDIntervals() / reference_rate;
			JND_Reference_Intervals_Positions.assign(reference_size, 0);
			for (int i = 0; i < reference_size; i++) {
				JND_Reference_Intervals_Positions[i] = tail_intervals ? (static_cast<int>(numberOfJNDIntervals()) - reference_size + i) : ((i + 1)*reference_rate - 1);
				//cout << "reference position[" << i << "]:" << JND_Reference_Intervals_Positions[i] << "\n";
			}
		}
	}

	//std::cout << "JND_Reference_Intervals_Positions test" << std::endl;
	//PrintFormat("JND_Reference_Intervals_Positions test=%d\n", JND_Reference_Intervals_Positions.size());
	if (JND_Reference_Intervals_Positions.size() > 0) {
		// here is the reference included in the input
		number_of_references = static_cast<int>(JND_Reference_Intervals_Positions.size());
		number_of_calculated = static_cast<int>(numberOfJNDIntervals()) - (Calculate_JND?1:0)*number_of_references;
		//PrintFormat("number_of_calculated=%d,number_of_references=%d,numberOfJNDIntervals=%d\n", number_of_calculated, number_of_references, numberOfJNDIntervals());
		JND_Serial_Intervals_Positions = std::vector<int>(numberOfJNDIntervals(), 0);
		//std::cout << "number of calculated position: " << number_of_calculated << ", num of references " << number_of_references << "\n";
		JND_Calculated_Intervals_Positions = std::vector<int>(number_of_calculated, 0);
		JND_Interval_To_Reference = std::vector<int>(number_of_calculated, 0);
		int ind_calculated_intervals = 0;
		for (int i = 0; i < numberOfJNDIntervals(); i++) {
			int relevant_reference = i*number_of_references / static_cast<int>(numberOfJNDIntervals());
			if (!checkVector(JND_Reference_Intervals_Positions, i)&& Calculate_JND) {
				JND_Serial_Intervals_Positions[i] = ind_calculated_intervals;
				JND_Calculated_Intervals_Positions[ind_calculated_intervals] = i;
				JND_Interval_To_Reference[ind_calculated_intervals] = JND_Reference_Intervals_Positions[relevant_reference];
				//std::cout << "calculated interval position [" << i << "],reference_index[" << relevant_reference << "]: map to reference position [" << JND_Interval_To_Reference[ind_calculated_intervals] << "]\n";
				ind_calculated_intervals++;
			} else {
				JND_Serial_Intervals_Positions[i] = relevant_reference;
			}
		}
	} else {
		//std::cout << "no JND_Reference_Intervals_Positions test" << std::endl;
		// test if there is white noise level
		number_of_calculated = static_cast<int>(numberOfJNDIntervals());
		if (vh->hasVariable("WN_Power")) {
			JND_WN_Additional_Intervals = vh->getValue<std::vector<float> >("WN_Power");
			JND_Interval_To_Reference = std::vector<int>(numberOfJNDIntervals(), 0);
			number_of_references = static_cast<int>(JND_WN_Additional_Intervals.size());
		}
		// here is the case that reference does not include in the input
		JND_Calculated_Intervals_Positions = std::vector<int>(number_of_calculated, 0);
		for (int i = 0; i < number_of_calculated; i++) {
			JND_Calculated_Intervals_Positions[i] = i;
			if (number_of_references > 0) {
				// it has white noise pointers to the end of the input + blocks
				JND_Interval_To_Reference[i] = number_of_calculated + i*(number_of_references / number_of_calculated);
			}
		}
	}
		//std::cout << "post JND_Reference_Intervals_Positions test" << std::endl;
	//PrintFormat("JND_Reference_Intervals_Positions Passed=%d\n", JND_Reference_Intervals_Positions.size());
	if (isWritingRawJND()) {
		
		std::vector<std::string> jnd_raw_name = std::vector<std::string>();
		if (JND_Calculate_AI()) jnd_raw_name.push_back("jnd_ai");
		if (JND_Calculate_RMS()) jnd_raw_name.push_back("jnd_rms");
		size_t buffer_size = jnd_raw_name.size()*numberOfJNDIntervals();
		size_t max_single_buffer_size = numberOfJNDIntervals() - JND_Reference_Intervals_Positions.size();
		if (buffer_size > JND_MAX_BUFFER_SIZE) buffer_size = JND_MAX_BUFFER_SIZE;
		if (vhout->Is_Matlab_Formatted() == 0) {
			vhout->setTargetsForWrite(Raw_JND_Target_File(), jnd_raw_name, new OutputBuffer<float>(Raw_JND_Target_File(), static_cast<long>(buffer_size), false));
		} else {
			// important note, since in mat ouput scheme all output unified, I will not want the buffer to be flushed when outputing to raw jnd file
			//auto char_matrix = createCharMatrix(jnd_raw_name);
			//vhout->setTargetsForWrite(output_name, jnd_raw_name, mxCreateStructMatrix(1, 1, static_cast<int>(char_matrix.size()), char_matrix.data()));
			size_t Mcolumn_size = (JND_Signal_Source ? 1 : JND_Column_Size())*targetJNDNoises();
			//std::cout << "Mcolumn_size = " << Mcolumn_size << std::endl;
			//std::cout << "JND_Column_Size() = " << JND_Column_Size() << std::endl;
			//std::cout << "max_single_buffer_size = " << max_single_buffer_size << std::endl;
			/**
			for (auto minor : jnd_raw_name) {
				vhout->preAllocateMinor(minor, output_name, Mcolumn_size, max_single_buffer_size / Mcolumn_size);
			}
			*/
		}
	}
	if (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS && Calculate_JND) {
		std::vector<std::string> jnd_name = std::vector<std::string>();
		jnd_name.push_back("jnd_final");
		if (vhout->Is_Matlab_Formatted() == 0) {
			vhout->setTargetsForWrite(JND_File_Name, jnd_name, new OutputBuffer<float>(JND_File_Name, JND_MAX_BUFFER_SIZE, false));
		} else {
			/**
			for (auto minor : jnd_name) {
				vhout->preAllocateMinor(minor, output_name, JNDInputTypes(), numberOfApproximatedJNDs() / JNDInputTypes());
			}
			*/
		}

	}
	//PrintFormat("Passed Params\n");
	if (vh->hasVariable("Database_Creation")) {
		database_creation = 1;
		strcpy_s<FILE_NAME_LENGTH_MAX>(database_file_name, vh->getValue<std::string>("Database_Creation").c_str());
		//printf("Database_Creation = %s !!!!\n",database_file_name);
	}
}

std::vector<float> CParams::loadSignalFromTagToVector(std::string file_tag,  int start_index, int end_index) {
	int nodes_number = end_index - start_index;
	TBin<double> *signalReader = new TBin<double>(vh->getValue<std::string>(file_tag), BIN_READ, true);
	cout << "reading " << vh->getValue<std::string>(file_tag) << " with " << nodes_number << " nodes\n";
	std::vector<double> Noise_double(nodes_number, 0.0);
	signalReader->read_padd(Noise_double, start_index, end_index);
	std::vector<float> v = castVector<double, float>(Noise_double);
	delete signalReader;
	return v;
}

int CParams::calculateBackupStage() {
	int backup_stage = -1;
	
	//std::cout << "Examines calculateBackupStage has TEST_File_Target? " << (vh->hasVariable("TEST_File_Target")>0) << "...." << std::endl;
	if (vh->hasVariable("TEST_File_Target")) {
		//std::cout << "testing type test output = " << Type_TEST_Output << ", stages_numbers.count(" << Type_TEST_Output << ")= " << stages_numbers.count(Type_TEST_Output) << std::endl;
		//std::string filename_test = params[params_set_counter].vh->getValue<std::string>("TEST_File_Target");
		if (stages_numbers.count(Type_TEST_Output)) {
			backup_stage = stages_numbers[Type_TEST_Output];
		}
	}
	return backup_stage;
}

void CParams::get_stage_data(std::vector<float>& v, int start_output_time_node_offset, int start_input_time_node_offset, int time_nodes, int sections) {
	std::vector<float> result;
	if (vh->hasVariable("Run_Stage_Vector")) {
		result = vh->getValue<std::vector<float> >("Run_Stage_Vector");
		auto st_input = result.begin() + (start_input_time_node_offset*sections);
		auto end_input = std::next(st_input, time_nodes*sections + 1);
		auto st_output = v.begin() + (start_output_time_node_offset*sections);
		std::copy(st_input, end_input, st_output);
	} else {
		TBin<float> *signalReader = new TBin<float>(vh->getValue<std::string>(Run_Stage_File_name), BIN_READ, true);
		//result = std::vector<float>(time_nodes*sections);
		signalReader->read_padd(v, start_output_time_node_offset*sections, start_input_time_node_offset*sections, time_nodes*sections);
		delete signalReader;
	}
}

void CParams::get_stage_data(std::vector<double>& v, int start_output_time_node_offset, int start_input_time_node_offset, int time_nodes, int sections) {
	std::vector<float> result;
	if (vh->hasVariable("Run_Stage_Vector")) {
		result = vh->getValue<std::vector<float> >("Run_Stage_Vector");
		auto st_input = result.begin() + (start_input_time_node_offset*sections);
		auto end_input = std::next(st_input, time_nodes*sections + 1);
		auto st_output = v.begin() + (start_output_time_node_offset*sections);
		std::copy(st_input, end_input, st_output);
	} else {
		result = std::vector<float>(time_nodes*sections, 0);
		TBin<float> *signalReader = new TBin<float>(vh->getValue<std::string>(Run_Stage_File_name), BIN_READ, true);
		//result = std::vector<float>(time_nodes*sections);
		signalReader->read_padd(result,0, start_input_time_node_offset*sections, time_nodes*sections);
		delete signalReader;
		auto st_input = result.begin();
		auto end_input = std::next(st_input, time_nodes*sections + 1);
		auto st_output = v.begin() + (start_output_time_node_offset*sections);
		std::copy(st_input, end_input, st_output);
	}
}