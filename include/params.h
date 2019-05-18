/******************************************************************\

	File Name :		params.h
	===========

	Classes Defined :	CParam
	=================

	Description	: 
	=============
		A complete model of the cochlea. The clas holds all parameters that are 
		fixed to every step (not changing in time);

\******************************************************************/
#pragma once

#ifndef __PARAMS
#define __PARAMS
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <sstream>
# include <iostream>
#include <locale>         // std::locale, std::tolower
#include <algorithm>
#include <cctype>
# include <cmath>
#include "cvector.h"
# include "const.h"
#include <bitset>
#include <limits>
# include <map>
# include <regex>
# include "IowaHillsFilters\IIRFilterCode.h"
# include "cvector.h"
# include "cochlea_common.h"
#include "cochlea_utils.h"
#include "ComplexJNDProfile.h"
#include "ConfigFileHandler.h"
#include "VirtualHandler.h"
#ifdef MATLAB_MEX_FILE
#include "MEXHandler.h"
#include <mexplus/mexplus.h>
#endif
/**
Inner parametrs structure
*/
class CParams  
{

	 

public:
int database_creation;
int sim_type;
int in_file_flag;
int input_noise_file_flag;
char in_file_name[FILE_NAME_LENGTH_MAX];
char in_noise_file_name[FILE_NAME_LENGTH_MAX];
int gamma_file_flag;
char gamma_file_name[FILE_NAME_LENGTH_MAX];
int nerve_file_flag;
int show_transient; // if 1 will show first Overlap time from doron algorithm, notice if duration divided to multiple interval, only first interval will include transition time
bool disable_lambda;
char nerve_file_name[FILE_NAME_LENGTH_MAX];
char database_file_name[FILE_NAME_LENGTH_MAX];
char backup_file_name[FILE_NAME_LENGTH_MAX];
char output_file_name[FILE_NAME_LENGTH_MAX];
char lambda_high_file_name[FILE_NAME_LENGTH_MAX];
char lambda_medium_file_name[FILE_NAME_LENGTH_MAX];
char lambda_low_file_name[FILE_NAME_LENGTH_MAX];
char default_lambda_high_file_name[FILE_NAME_LENGTH_MAX];
char default_lambda_medium_file_name[FILE_NAME_LENGTH_MAX];
char default_lambda_low_file_name[FILE_NAME_LENGTH_MAX];
char ac_filter_file_name[FILE_NAME_LENGTH_MAX];
long Fs; // the sample frequency default SAMPLE_RATE
float scaleBMVelocityForLambdaCalculation;
int Show_Device_Data; // if 1 show full cuda device data, 0 otherwise
/**
* this parameters controls error boundaries on BM calculation procedure
* modify carefully
*/
float Max_M1_SP_Error_Parameter;
float Max_Tolerance_Parameter;
int Relative_Error_Parameters;
float M1_SP_Fix_Factor;
float Tolerance_Fix_Factor;

int Show_Calculated_Power;// for boundaries
/** 
*	number of time blocks for the cuda to handle in parrallel
*	for example with 16 time blocks each 20ms default, algorithm will handle
*	0.32 seconds of sound at each interval
*/
int Time_Blocks; 


/** 
*	length of time block in seconds, BM computing algorith
*	must be at least 0.02s, advanced feature.
*	higher number will cause algorithm to lower parrarllelism
*/
float Time_Block_Length; 
// time block length in microseconds
size_t Time_Block_Length_Microseconds;

float sin_freq;
float sin_dB;
float sin_amp;
float duration;
double SPLRefVal;
std::vector<float> ihc_vector;
std::vector<float> ohc_vector;
float cuda_max_time_step; // max time advancement in cuda run default 1e-6
float cuda_min_time_step; // max time advancement in cuda run default 1e-14
bool Show_JND_Configuration; // for debugging JND array sizes
bool disable_advanced_memory_handling;
bool run_ihc_on_cpu;
int ihc_mode;
int ohc_mode;
int num_frequencies; // number of pure pitch tone signals processed
int frequncies_analysis;
int continues; // if 0 will not attempt analyze the entire file if too long
int loadPrevData; // will load prev results from disk in order to create continuty in lambda
float frequncies[100]; // power of input for each frequency to thest in dBSPL
float offset; // start off set of analyzing the file in seconds
float Lambda_SAT; // lambda saturation default RSAT
float Tdelta; // the dc filter time default Tdc
float eta_AC; // IHC AC coupling[V / s / cm]
float eta_DC; // IHC DC coupling[V / cm]
size_t overlapTimeMicroSec;  // overlap time of blocks in micro seconds
double overlapTime; // over lap time
double Processed_Interval; // float represents the length in time of input processed used to calculate time blocks
int JACOBBY_Loops_Fast; // number of jcoby loops to perform on fast approximation
int JACOBBY_Loops_Slow; // number of jcoby loops to perform on slow approximation	
int Cuda_Outern_Loops; // max control loops
int JND_Signal_Source; // 0 for normal sin test, 1 - for reading signal from either Input_Signal or Input_File
/**
*	ac filter params for functionality
*/
int ac_filter_mode; // 0 for file, 1 for function
char ac_filter_params[FILE_NAME_LENGTH_MAX]; // string of params in format of name1=value1,name2=value2
float Fc; // Frequency cut for lpf
float Fpass; // Fpass for lpf
float Fstop; // Sstop for lpf
float Apass; // gain at Fc in DB
float Astop; // Attenuation stop
float Wpass; // ripple in passband
float Wstop; // weight of stop in equiripple
float Slope; // slope currently unsupported I dont know what to do with it
int FilterOrder; // filter order
int Decouple_Unified_IHC_Factor;
butterMatch butterType;
std::string filterName; // filter name like Butterworth, should be lpf
std::string mode; // additional features of filter
bool show_filter; // if true show generated filter
bool deltasFound; // if true ripple pass and ripple stop found and than will calculate weights and order from there
bool minOrder; // if true will override order and use Oppenheim & Schafer 2nd addition DSP formula 7.104 to calculate mininmum order
int Show_Run_Time; // array of flags to  show run time for each cuda function run
int Show_CPU_Run_Time; // array of flags to show run time for CPU sub routines
unsigned int Allowed_Outputs; // flag array for allowed outputs 0 bit for BM velocity and 1-3 are lambda high to lambda low accordingliy
// JND Parameters
bool Calculate_JND;
unsigned int Convergence_Tests;
unsigned int Calculate_JND_Types; // bit flag of jnd types to calculate, 1 is mean square, 2 is All Informatin
bool Calculate_JND_On_CPU; // force JND calculation on CPU
float JND_Interval_Duration; // interval in seconds for input to be processed for JND
float JND_Ignore_Tail_Output; // if positive  will ignore last samples at this duration for JND calculations, will get overlap time on default
std::vector<int> JND_Reference_Intervals_Positions;	// index of referenced intervals
std::vector<int> JND_Calculated_Intervals_Positions; // index of calculated interval
std::vector<int> JND_Serial_Intervals_Positions; // indexes in calculated and reference from position on the input
std::vector<int> JND_Interval_To_Reference;// for each index as calculated interval value is reference index, references point to themselfs
std::vector<float> JND_WN_Additional_Intervals; // add white noise blocks of reference, to the input
std::vector<double> Input_Signal; // if input signal included it will be read here
std::vector<double> Input_Noise; // if input noise included for complex generation it will be read here
float JND_Interval_Head; // interval in seconds to ignore when calculate JND from the beggining of the JND interval time
float JND_Interval_Tail; // interval in seconds to ignore when calculate JND from the end of the JND interval time
std::string JND_File_Name; // target file to write JND into
std::string JND_Raw_File_Name; // in case of complex jnd calculation will write middle results here
bool hasJND_Raw_Output; // true if jnd raw file name loaded
std::vector<float> dA_JND; // max power for each input interval
std::vector<float> W; // the W parameter in JND calculation, weight of importance for groups of Nerves
std::vector<float> Aihc; // the A parameter in lambda default 
std::vector<float> spontRate; // spontanus rate of spikes per group of nerves 
std::string output_name; // output target file
std::string Signal_Name; // input signal name if read from file
std::string Noise_Name; // noise file name if read from file
std::vector<device_jnd_params> inputProfile; // preserve power (and frequency for each signal analyzed for JND)
std::vector<ComplexJNDProfile> complexProfiles; // preserve parametrs for each signal
bool showGeneratedConfiguration; // if true will print parametrs for jnd calculations
int Show_Generated_Input; // if 1 will output generated input
int JND_USE_Spont_Base_Reference_Lambda; // if 0 will remove spont rate from lambda when calculationg JND
int JND_Noise_Source; // 0 - self generate, 1 - read from input file
int BMOHC_Kernel_Configuration;
std::vector<std::string> BMOHC_Kernel_Configurations_Names;
float maxdBSPLJND; // maximum value forspl measurements
std::string Type_TEST_Output;// type of output
/**
* for generate complex input from profiles for fast JND tests
*/
std::vector<float> testedFrequencies; // tested frequencies in HZ
std::vector<float> testedPowerLevels; // tested power levels in dB SPL Ref related to SPLRefVal, 1111 for no power (-Inf)
std::vector<float> testedNoises; // tested Noises levels  in dB, 1111 for no power
std::vector<float> Hearing_AID_FIR_Transfer_Function; // Hearing aid B array of transfer function
std::vector<float> Hearing_AID_IIR_Transfer_Function; // Hearing aid A array of transfer function
std::vector<double> AC_Filter_Vector; // raw data to process filter vector
float Approximated_JND_EPS_Diff; // approximated jnd epsilon diff, used in complex approximation of JND only will be half diff of power levels or overrided
bool generatedInputProfile;
std::vector<float> M_tot; // total number of fibers default 30000
// review bugs parameters set names in config file to 1 to activate




bool Review_Lambda_Memory;
bool Review_AudioGramCall;
bool generaldebug;
unsigned int Examine_Configurations; // use to view configurations on the fly
unsigned int Complex_JND_Calculation_Types; // bit array in case of complex jnd aggregation 1 - is for minimu calculation, 2 - for wanted type
unsigned int IS_MEX; // 1 if its mex file
unsigned int JND_Include_Legend; // bit array 1 for final, 2 for rms,4 for AI
int Run_Fast_BM_Calculation; // will run BM calculation with relaxed memory requirements
int Decouple_Filter; //  if 0 it will use backward calculating for each block except the first, otherwise it will decouple the input every [Decouple_Filter] number of time blocks
bool Generate_One_Ref_Per_Noise_Level; // if true will generate one reference interval per noise level instead of power level
bool View_Complex_JND_Source_Values; // if true will show complex jnd values for minimum searching
bool Calculate_From_Mean_Rate; // if true (default will calculate final jnd from mean rate, else will calculate it from all information
bool Discard_BM_Velocity_Output; // if true do not write bm velocity to disk
bool Discard_Lambdas_Output; // if true do not write Lambda to disk
bool Perform_Memory_Test;
double JND_Delta_Alpha_Time_Factor;   // time for cramer raw factor,0 for default
std::vector<double> Noise_Expected_Value_Accumulating_Boundaries;
std::vector<double> Noise_Sigma_Accumulating_Boundaries;
//std::map<std::string, std::string> paramsMap;
std::map<std::string, std::string> filtersMap; // contains data of filter function
std::map<std::string, std::string> filtersMapRaw; // contains data of filter with values and keys un lowered
std::map<std::string, std::string> filtersKeysStat; // map of lowered to uppear keys
std::map<std::string, int> stages_numbers; // for debug stages
VirtualHandler *vh; // handler of input file
VirtualHandler *vhout; // handler of output file
//VirtualHandler parser; // parser of input
double Remove_Generated_Noise_DC; // if 1 removes dc from generated noise
int Normalize_Sigma_Type; // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
int Normalize_Sigma_Type_Signal; // 0 - normalize sigma to 1, 1 - normalize sigma summary to 1, 2 - normalize sigma summary to given time interval at Normalize_Noise_Energy_To_Given_Interval
double Normalize_Noise_Energy_To_Given_Interval; // normalizes noise energy to give time in seconds, default 0.16
int Remove_Transient_Tail; // if transient exist will remove the overlap tail from the end of the output, ignore otherwise
int Filter_Noise_Flag; // 1- to filter the noise signal
std::vector<std::string> Verbose_BM_Vectors;
std::string Filter_Noise_File; // encoded filter file if there is noise flag
int Run_Stage_Calculation; // 0 run from regular input, 1 run from BM velocity stage, 2 run from lambda stage // will need appropriate input array, as float
std::string Run_Stage_File_name; // file name to read run stage if necessary

CParams(void);
CParams(unsigned int);// for allocating as MEX
~CParams(void);
// functions
CParams& operator=(const CParams& p);
void parse_parameters_file(char *pfname, bool Review_Parse);
void parse_params_map(bool Review_Parse); // used to parse the param map after loaded from file, set review parse to true to view parsing data
 void parse_arguments_performance(char* argv[], int &position,int argc);
//int get_two_tokens(char *lline, char *param, char *val);
//void get_line(FILE *fp, char *lline);
bool advanceOnNotNumeric(char* argv[], int &position,int argc); // if true its next config aborting
bool notEmptyString(char* str);
bool notEmptyString(const std::string& str);
std::vector<double> Complex_Profile_Noise_Level_Fix_Factor; // will multiplicate this factor of noise power to signal power in case of a noise  
std::vector<double> Complex_Profile_Noise_Level_Fix_Addition; // will add this this factor of noise power to signal power in case of a noise
std::vector<double> Complex_Profile_Power_Level_Divisor;
std::string getNoiseName();
std::string getSignalName();
int Mex_Debug_In_Output;
inline int forceMexDebugInOutput() { return Mex_Debug_In_Output*IS_MEX;  }
inline const int Has_Input_Signal() const { return !Input_Signal.empty();  }
inline const int Has_Input_Noise() const { return !Input_Noise.empty(); }
inline const float MAX_M1_SP_ERROR_Function() const { return Max_M1_SP_Error_Parameter; }
inline const float MAX_Tolerance_Function() const { return Max_Tolerance_Parameter; }
inline const float intervalTime() const { return Time_Block_Length*Time_Blocks; }
inline const double Ts() const { return (1.0 / static_cast<double>(Fs)); }
inline const size_t calcTimeBlockNodes() const { return static_cast<size_t>(ceil(Time_Block_Length*Fs)); }
inline const size_t calcOverlapNodes() const { return static_cast<size_t>(overlapTimeMicroSec*Fs / 1000000); }
//int calcLastSavedTimeBlock(); // calculating the	equivalent fo SAMPLE_BUFFER_LEN_SHORT dependendant on input parameters
inline const float intervalTimeBlock() const { return Time_Block_Length; }
inline size_t numberOfJNDIntervalsRawInput() { return static_cast<int>(roundf(duration / JND_Interval_Duration)); }
inline size_t numberOfJNDIntervalsComplexGeneration() {
	return static_cast<size_t>(inputProfile.size());
}
inline int numberOfJNDIntervalsComplexGenerationRawCalculation() { return static_cast<int>(testedNoises.size())*((Generate_One_Ref_Per_Noise_Level&&Calculate_JND ? 1 : 0) + (static_cast<int>(testedPowerLevels.size())*(static_cast<int>(JNDInputTypes()) + (Generate_One_Ref_Per_Noise_Level||!Calculate_JND ? 0 : 1)))); }

inline int JNDInputTypes() { return JND_Signal_Source ? 1 : static_cast<int>(testedFrequencies.size()); }
inline size_t targetJNDNoises() { return testedNoises.empty() ? 1 : testedNoises.size(); }
inline int 	Noise_Expected_Value_Accumulating_Start_Index() { return Noise_Expected_Value_Accumulating_Boundaries.size() > 0 ? static_cast<int>(Noise_Expected_Value_Accumulating_Boundaries[0]*Fs) : static_cast<int>((JND_Interval_Head)*Fs); }
inline int 	Noise_Expected_Value_Accumulating_End_Index() { return Noise_Expected_Value_Accumulating_Boundaries.size() > 1 && Noise_Expected_Value_Accumulating_Boundaries[1]>0 ? static_cast<int>(Noise_Expected_Value_Accumulating_Boundaries[1] * Fs) : static_cast<int>((JND_Interval_Duration - JND_Interval_Tail)*Fs); }
inline int 	Noise_Sigma_Accumulating_Start_Index() { return Noise_Sigma_Accumulating_Boundaries.size() > 0 ? static_cast<int>(Noise_Sigma_Accumulating_Boundaries[0] * Fs) : static_cast<int>((JND_Interval_Head)*Fs); }
inline int 	Noise_Sigma_Accumulating_End_Index() { return Noise_Sigma_Accumulating_Boundaries.size() > 1 && Noise_Sigma_Accumulating_Boundaries[1]>0 ? static_cast<int>(Noise_Sigma_Accumulating_Boundaries[1] * Fs) : static_cast<int>((JND_Interval_Duration - JND_Interval_Tail)*Fs); }
inline std::bitset<std::numeric_limits<unsigned int>::digits> getTestConvergence() { return std::bitset<std::numeric_limits<unsigned int>::digits>(Convergence_Tests); }
inline std::bitset<std::numeric_limits<unsigned int>::digits> JNDComplexMethods() { return std::bitset<std::numeric_limits<unsigned int>::digits>(Complex_JND_Calculation_Types); }
// allowed outputs from BM velocity and Lambdas High,Medium and Low
inline std::bitset<std::numeric_limits<unsigned int>::digits> Allowed_Output_Flags() { return std::bitset<std::numeric_limits<unsigned int>::digits>(Allowed_Outputs); }
inline std::bitset<std::numeric_limits<unsigned int>::digits> GetExamineConfigurationsFlags() { return std::bitset<std::numeric_limits<unsigned int>::digits>(Examine_Configurations);  }
inline bool Examine_Configurations_Flags(int output_number) { return GetExamineConfigurationsFlags().test(output_number); }
inline bool Convergence_Test_Flags(int output_number) { return getTestConvergence().test(output_number); }
inline bool Allowed_Output_Flags(int output_number) { return Allowed_Output_Flags().test(output_number); }
inline size_t numberOFJNDComplexMethods() { return JNDComplexMethods().count(); }
inline size_t targetJNDComplexProfiles() { return targetJNDNoises()*JNDInputTypes(); }

inline int numberOfApproximatedJNDsPerMethod() { return sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS ? static_cast<int>(JNDInputTypes()*testedNoises.size()) : 0; } // summary of jnd interval after approximation algorithm
inline int numberOfApproximatedJNDs() { return sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS ? static_cast<int>(JNDInputTypes()*testedNoises.size()*numberOFJNDComplexMethods()) : 0; } // summary of jnd interval after approximation algorithm
inline size_t numberOfJNDIntervals() { return sim_type == SIM_TYPE_VOICE ? numberOfJNDIntervalsRawInput() : numberOfJNDIntervalsComplexGeneration(); }

inline int positiveDecoupler() { return Decouple_Filter == 0 ? 1 : Decouple_Filter; }
inline bool includeJNDReference() { return JND_Reference_Intervals_Positions.size() > 0; }
inline int JND_Interval_Nodes_Offset() { return static_cast<int>(roundf(Fs*JND_Interval_Head)); }
inline int JND_Interval_Nodes_Full() { return static_cast<int>(roundf(Fs*JND_Interval_Duration)); }
inline int JND_Interval_Nodes_Length() { return static_cast<int>(roundf(Fs*(JND_Interval_Duration - JND_Interval_Head - JND_Interval_Tail))); }
inline double JND_Delta_Alpha_Length_Factor() { return JND_Delta_Alpha_Time_Factor == 0 ? (1.0*static_cast<double>(JND_Interval_Nodes_Length())) : JND_Delta_Alpha_Time_Factor*static_cast<double>(Fs); }

inline double calculatedDurationGenerated() { return static_cast<double>(positiveDecoupler()*Time_Block_Length*numberOfJNDIntervalsComplexGenerationRawCalculation()); }
inline bool JND_Type_Flag_Test(unsigned int flagPosition) { 
	//cout << "(Calculate_JND_Types&(1 << flagPosition))\n(" << (Calculate_JND_Types) << "&" << (1 << flagPosition) << ")\n" << (Calculate_JND_Types&(1 << flagPosition))<<"\n";
	return (Calculate_JND_Types&(1 << flagPosition)) > 0;
}
inline bool JND_Calculate_RMS() { return JND_Type_Flag_Test(0);  }
inline bool JND_Calculate_AI() { return JND_Type_Flag_Test(1); }
inline bool Generating_Input_Profile() { return sim_type == SIM_TYPE_PROFILE_GENEARATING || sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS; }
void updateInputProfile();
inline double SPL2PA(double dBSPL) { return dBSPL == MIN_INF_POWER_LEVEL?0.0:(10.0 * pow(10.0, dBSPL / 20.0)*SPLRefVal); }
inline double Pa2SPL(double Pa) { return 20 * log10(Pa / (10 * SPLRefVal)); }
inline double Pa2SPLForJND(double Pa) { return 20.0 * log10(Pa / SPLRefVal); }
inline double SPL2PAForFinalJND(double dBSPL) { return dBSPL == MIN_INF_POWER_LEVEL ? 0.0f : (pow(10.0, dBSPL / 20.0)*SPLRefVal); }
inline double maxPAForFinalJND() { return SPL2PAForFinalJND(maxdBSPLJND); } // to test minimum level of pa apply
inline double SPL2PAForFinalJNDBounded(double Pa) { return isinf(Pa)||(1.0/Pa) > maxPAForFinalJND() ? maxdBSPLJND : Pa2SPLForJND(1.0 / Pa); } // to test minimum level of pa apply
inline double PA2SPLForFinalJNDBounded(double Pa) { return isinf(Pa) || Pa > maxPAForFinalJND() ? maxdBSPLJND : Pa2SPLForJND(Pa); }
inline int profilesPerRun(int from_profile_index) {
	int ppr = 0;
	//cout << "inputProfile.size()<" << inputProfile.size() << ">*JND_Interval_Duration<" << JND_Interval_Duration << "> <= intervalTime()<" << intervalTime() << ">\n";
	if (static_cast<int>(inputProfile.size())*JND_Interval_Duration <= intervalTime()) {
		ppr = static_cast<int>(inputProfile.size());
	} else {
		ppr = static_cast<int>(round(intervalTime() / JND_Interval_Duration));
		ppr = __tmin(static_cast<int>(inputProfile.size()) - from_profile_index, ppr);
	}
	return ppr;
}
inline int IntervalDecoupledRemainder() { return Decouple_Filter > 1 ? Time_Blocks%Decouple_Filter : 0; }
inline int IntervalDecoupled() { return Decouple_Filter == 1 || (Decouple_Filter > 1 && IntervalDecoupledRemainder() == 0) ? 1 : 0; }
inline bool isWritingRawJND() { return Calculate_JND && (hasJND_Raw_Output || sim_type != SIM_TYPE_JND_COMPLEX_CALCULATIONS); }
inline size_t overlapnodesFixReduce(const double& start_time) { return  show_transient == 0 || Remove_Transient_Tail == 1 || (((IntervalDecoupled() && start_time < duration) || (Decouple_Filter == 0 && start_time > intervalTime()))) ? calcOverlapNodes()*SECTIONS : 0; }
inline string Raw_JND_Target_File() { return isWritingRawJND() ? (sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS?JND_Raw_File_Name:JND_File_Name) : ""; }
string showJNDConfiguration(const device_jnd_params& config, const int& index);
//inline float intervalTimeBlockExtended() { return Time_Block_Length + static_cast<float>(Ts())*(calcLastSavedTimeBlock()/Time_Blocks); }
/*
inline float intervalTimeBlockExtendedOverlapped() {
	return static_cast<float>(intervalTimeBlockExtended() + overlapTime);
	//return intervalTimeBlockExtended() + TIME_OFFSET_NORMALIZED;
}
*/
inline void addStageNumber(const std::string& str, const int value) {
	//std::cout << "addStageNumber("<<str<<","<<value<<")" << std::endl;
	stages_numbers.insert(std::pair<std::string, int>(str, value));
}
inline void setOverlapTime(double value) {
	overlapTime = value;
	JND_Ignore_Tail_Output = static_cast<float>(overlapTime);
	overlapTimeMicroSec = toMicroSeconds(overlapTime);
}

inline const float intervalTimeBlockOverlapped() const {
	return static_cast<float>(intervalTimeBlock() + overlapTime);
	//return intervalTimeBlock() + TIME_OFFSET_NORMALIZED;
}
inline const size_t toMicroSeconds(double sec) const { return static_cast<size_t>(round(sec * 1000) * 1000); }
inline const size_t toMicroSeconds(float sec) const {
	return toMicroSeconds(double(sec));
}
inline size_t JND_Column_Size() { return testedPowerLevels.empty() ? 1 : testedPowerLevels.size(); }
inline double toSeconds(long microsec) { return double(microsec) / 1000000; }
//inline int calcTimeBlockExtendedNodes() { return calcTimeBlockNodes() + static_cast<int>(ceil(static_cast<float>(calcLastSavedTimeBlock()) / static_cast<float>(Time_Blocks))); }
inline int cudaIterationsPerKernel() { return static_cast<int>(1.0f / (Fs*cuda_max_time_step)); }
// output will always include transient, show transient itself will check whether to copy transient to hard drive too
inline const size_t totalTimeNodes() const { return calcOverlapNodes()+calcTimeBlockNodes() *Time_Blocks; } // total time nodes without last block saved
//inline int totalTimeNodesExtended() { return calcOverlapNodes() + calcTimeBlockExtendedNodes() *Time_Blocks; } // total time nodes with last block saved
//inline int totalTimeNodesExtendedP1() { return calcOverlapNodes() + calcTimeBlockExtendedNodes() *(Time_Blocks + 1); } // total time nodes+1 with last block saved
inline const size_t totalTimeNodesP1() const { return calcOverlapNodes() + calcTimeBlockNodes() *(Time_Blocks + 1); } // total time nodes+1 with last block saved
inline const size_t totalResultNodes() const { return totalTimeNodes() *SECTIONS; } // total result nodes without last block saved
inline const int saveLambda() const { return (Allowed_Outputs>0 || Calculate_JND_On_CPU)?1:0; }
inline int fullBuffer() { int bstage = calculateBackupStage(); return (bstage==8||bstage==3); }
inline int allocateFullBuffer() { return saveLambda() || fullBuffer(); }
/**
* Contains start stage to run from, can start from input, after BM velocity or after lambda calculation
*/
inline const int Run_Stage_Verified() const { return Run_Stage_Calculation == 0 || (!Run_Stage_File_name.empty()||vh->hasVariable("Run_Stage_Vector")); }
inline const int Run_Stage_Unverfied() const { return Run_Stage_Calculation > 0 && !Run_Stage_Verified(); }
inline const int Run_Stage_Input() const { return Run_Stage_Calculation == 0; }
inline const int Run_Stage_BM_Velocity() const { return Run_Stage_Calculation == 1 && Run_Stage_Verified();  }
inline const int Run_Stage_Before_Lambda() const { return Run_Stage_Calculation < 2 && Run_Stage_Verified(); }
inline const int Run_Stage_Lambda() const { return Run_Stage_Calculation == 2 && Run_Stage_Verified(); }
void get_stage_data(std::vector<float>& v, int start_output_time_node_offset, int start_input_time_node_offset, int time_nodes, int sections);
void get_stage_data(std::vector<double>& v, int start_output_time_node_offset, int start_input_time_node_offset, int time_nodes, int sections);
inline void get_stage_data(std::vector<float>& v, int start_input_time_node_offset, int time_nodes, int sections) {
	get_stage_data(v, 0, start_input_time_node_offset,time_nodes,sections);
}
inline void get_stage_data(std::vector<double>& v, int start_input_time_node_offset, int time_nodes, int sections) {
	get_stage_data(v, 0, start_input_time_node_offset, time_nodes, sections);
}

std::vector<float> loadSignalFromTagToVector(std::string tag_name, int start_index, int end_index);
int calculateBackupStage();
#ifdef CUDA_MEX_PROJECT
void parseMexFile(const mxArray *input_arrays[], mxArray *output_arrays[]);
#endif
};
# endif