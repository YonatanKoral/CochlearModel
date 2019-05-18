#pragma once

#ifndef __AUDIOGRAMCREATOR
#define __AUDIOGRAMCREATOR
#include "const.h"
#include "params.h"
#include "cvector.h"
#include "cochlea_common.h"
#include "model.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <bitset>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
/** additions to support cuda procedures here */
#include "cochlea_common.h"
#include"OutputBuffer.h"
#include "smaller_than.h"
#include "ComplexJNDProfile.h"
#include "Log.h"
/** final special additions */

extern template void GeneralKernel_Copy_Results_Template<double>(double *target,double *src, size_t size);
extern template void GeneralKernel_Copy_Results_Template<float>(float *target, float *src, size_t size);
extern template void GeneralKernel_Copy_Results_Template<double>(double *target, double *src, size_t size, size_t offset);
extern template void GeneralKernel_Copy_Results_Template<float>(float *target, float *src, size_t size, size_t offset);
extern template void ReverseKernel_Copy_Results_Template<float>(float *cpu_src, float *cuda_target, size_t start_time_node, size_t time_nodes, int sections);
extern template void ReverseKernel_Copy_Results_Template<double>(double *cpu_src, double *cuda_target, size_t start_time_node, size_t time_nodes, int sections);
extern template void updateCUDALambdaArray<JNDFloat>(JNDFloat *lambda_array, JNDFloat *cuda_buffer, size_t calc_time_nodes, int sections, int Show_Run_Time, int Show_Device_Data, int cuda_buffer_update, Log &outer_log);
class AudioGramCreator
{
public:
	size_t time_dimension;
	size_t write_time_dimension;
	int sections;
	int filter_size;
	int time_blocks;
	int DC_filter_size;
	float start_time;
	//float spont[LAMBDA_COUNT];
	//float A[LAMBDA_COUNT];
	//float W[LAMBDA_COUNT];
	std::vector<double> filter_dc;
	double filter_a[DEVICE_MAX_FILTER_ORDER];
	double filter_b[DEVICE_MAX_FILTER_ORDER];
	HFunction tffull;
	std::vector<int> Failed_Converged_Blocks;
	std::vector<int> Failed_Converged_Signals;
	std::vector<int> Failed_Converged_Summaries;
	std::vector<float> cudaTestBuffer;
	std::vector<lambdaFloat> lg10;
	std::vector<float> BM_input;
	std::vector<float> original_ihc;
	std::vector<lambdaFloat> AC;
	std::vector<lambdaFloat> DC;
	std::vector<lambdaFloat> IHC;
	std::vector<double> IHC_damage_factor;
	std::vector<float> Lambda;
	std::vector<JNDFloat> dLambda;
	std::vector<lambdaFloat> cudaLambdaFloatBuffer;
	std::vector<lambdaFloat> cudaLambdaFloatShortBuffer;
	std::vector<JNDFloat> dSquareLambda;
	std::vector<JNDFloat> dSumLambda;
	std::vector<float> buffer;
	std::vector<lambdaFloat> Shigh;
	std::vector<lambdaFloat> dS;
	std::vector<JNDFloat> MeanRate;
	std::vector<JNDFloat> AvgMeanRate; // for testing of cuda sub threshold SNR
	std::vector<JNDFloat> dMeanRate;
	std::vector<JNDFloat> dSquareMeanRate;
	std::vector<double> nIHC;
	std::vector<JNDFloat> CRLB_RA;
	std::vector<JNDFloat> preFisherAITimeReduced;
	std::vector<JNDFloat> preFisherAI;
	std::vector<JNDFloat> JND_RA;
	std::vector<JNDFloat> FisherAI;
	std::vector<JNDFloat> F_RA;
	std::vector<JNDFloat> FisherAISum;
	std::vector<double> RateJNDall;
	std::vector<double> AiJNDall;
	std::vector<double> ApproximatedJNDall;
	std::vector<int> ApproximatedJNDallWarnings;
	std::vector<float> gamma;
	std::vector<double>  dA;
	float Nerves_Clusters[3 * LAMBDA_COUNT];
	CParams *params;
	CModel *model;
	Log audiogramlog;
	//float ANno;
	//float PnIHC;
	int handeledIntervalsJND; // number of handeled jnd intervals on this input
	int params_set_counter;
	size_t mean_size;
	size_t fisher_size;
	size_t allocate_time;
	float *backup_speeds;
	bool is_filter_fir;
	bool is_first_time_for_set;
	//void preapareBM(void);
	void readFilter(const HFunction& rawFilterData);
	void saveStageToDisk(float *,int,int,int,const char *,bool,bool);
	void saveArrayToDisk(float *,int,int,const char *,bool,bool);
	//void loadArrayFromDisk(float *subject,int array_length,const char *file_name);
	void saveFloatArrayToDisk(const std::vector<float>& subject, int, int, const char *, bool, bool);
	void loadFloatArrayFromDisk(float *subject,int array_length,const char *file_name);
	void loadOutputArrayFromDisk(float *subject,int array_length,int chunk_size,const char *file_name,bool include_time);
	void calcdA(std::vector<float>& input); // calcs dA for JND
	//void loadSavedSpeedsFromDisk();
	//bool isLoadingBackupSpeedsFromDisk();
	std::vector<std::string> getLegends(const int&);
	std::string getSimpleLegend(const int& isRMS, const int& interval);
	std::vector<std::string> getSimpleLegends(const int& isRMS, const std::vector<int>& intervals);
	void loadSavedSpeeds(std::vector<float>& input);
	// backups few last saved speeds for next round if necessary for reloads
	//void pointToBackupMS(float *);
	//void backupFewMS();
	//void  backupFewMSToDisk();
	//void filter(float *,float *,float *,float *,int,int);
	void ac_filter(void);
	//void vectorSum(float *A,float a,float *B,float b,float *C); // a*A - b*B == C
	void setupIHCOHC();
	void IHCCalc();
	void calcLambda(int lambda_index);
	void calcLambdas(Log &outer_log);
	void calcSum(float *src,float *dst,int cells,int jump_cells); // for untransposed coordinates will jump by sections
	void calcAvg(float *src,float *dst,int cells,int jump_cells); // for untransposed coordinates will jump by sections
	void calcJND(Log &outer_log); // calcs JND on CPU
	//void calcMeanRate();
	void saveLambdas(size_t overlap_reduce, size_t overlap_offset, Log &integrated_log);
	void calcJNDFinal(); // final stage to calculate on CPU
	void calcComplexJND(const vector<double>& values);
	inline void calcComplexJND() { calcComplexJND(RateJNDall); }
	//void square(float*,float *);
	//void FIRFilter(float *,float *,float *,int,int);
	void createDS();
	void filterDC();
	//void saveToDatabase();
	void runInCuda(float *host_bm_velocity, Log &outer_log);
	bool isRunIncuda(void);
	size_t matrixSize() const { return time_dimension*sections; }
	size_t writeMatrixSize() const { return write_time_dimension*sections; }
	//int timeIgnoredMatrix() const { return (matrixSize()-writeMatrixSize()); }
	size_t meanSize() const { return time_blocks*sections; }
	size_t allMeanSize() const { return LAMBDA_COUNT*meanSize(); }
	int lambdaOffset() { 
		return 0;
		//return static_cast<int>(0.004*params[params_set_counter].Fs);
	}
	//void calcFisher();
	void setupCreator(int, int, int, int, CParams*, CModel *, double, bool, int, std::vector<float>& input);
	void valuesSetupCreator(int, int, int, CParams*, CModel *, double, int, std::vector<float>& input);
	void freeAll();
	AudioGramCreator(void);
	AudioGramCreator(int, int, int, int, CParams*, CModel *, double, std::vector<float>& input);
	~AudioGramCreator(void);


	int lambda_nodes =0; // to remember #lambda nodes
	inline void setLambdaNodes(int nodes) { lambda_nodes = nodes; }
	inline const int getLambdaNodes() const { return lambda_nodes; }

	int BM_velocity_nodes = 0; // to remember #lambda nodes
	inline void setBMVelocityNodes(int nodes) { BM_velocity_nodes = nodes; }
	inline const int getBMVelocityNodes() const { return BM_velocity_nodes; }

	int mean_nodes = 0; // to remember #lambda nodes
	inline void setMeanNodes(int nodes) { mean_nodes = nodes; }
	inline const int getMeanNodes() const { return mean_nodes; }


	int fisher_nodes = 0; // to remember #lambda nodes
	inline void setFisherNodes(int nodes) { fisher_nodes = nodes; }
	inline const int getFisherNodes() const { return fisher_nodes; }


	int fisher_intervals = 0; // to remember #lambda nodes
	inline void setFisherIntervals(int nodes) { fisher_intervals = nodes; }
	inline const int getFisherIntervals() const { return fisher_intervals; }

	int approximated_jnd_intervals = 0; // to remember #lambda nodes
	inline void setApproxJNDIntervals(int nodes) { approximated_jnd_intervals = nodes; }
	inline const int getApproxJNDIntervals() const { return approximated_jnd_intervals; }
};

#endif