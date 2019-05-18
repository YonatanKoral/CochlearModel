#include "AudioGramCreator.h"
void AudioGramCreator::readFilter(const HFunction& rawFilterData) {
	//printf("opened filter file %s\n",file_name);
	tffull = rawFilterData;
	filter_size = static_cast<int>(rawFilterData.Numerator.size());

	is_filter_fir = rawFilterData.isFIR();
	//printf("loading from AudioGramCreator vector size is %.0f\n",vector_size_x[0]);
	for (int i=0;i<filter_size;i++) {
		if (!is_filter_fir)  filter_a[i] = rawFilterData.Denominator[i];
		// ac filter b array also give me the factor for convert from cm/s to m/s here
		filter_b[i] = rawFilterData.Numerator[i]; // params[params_set_counter].scaleBMVelocityForLambdaCalculation*static_cast<float>(rawFilterData.Numerator[i]);
		//printf("b[%d]=%.2e\n",i,filter_b[i]);
		//printf("filter_a[%d]=%.3e,filter_b[%d]=%.3e\n",i,filter_a[i],i,filter_b[i]);
	}
	/*
	for (int i=0;i<filter_size;i++){
		printf("a[%d]= %.4e; b[%d]= %.4e;\n",i,filter_a[i],i,filter_b[i]);
	}
	*/
}
void AudioGramCreator::saveStageToDisk(float *subject,int transposed_length,int written_length,int chunk_size,const char *file_name,bool include_time=true,bool reset_file=true) {
	AudioGramCreator::saveArrayToDisk(subject+transposed_length-written_length,written_length,chunk_size,file_name,include_time,reset_file);
}
void AudioGramCreator::saveArrayToDisk(float *subject,int array_length,int chunk_size,const char *file_name,bool include_time=true,bool reset_file=true) {
	int i;
	FILE *pfp;
	pfp = NULL;
	//cout << "saveArrayToDisk: " << file_name << " to write " << (array_length*sizeof(float)) << " bytes reset? " << reset_file << endl;
	float t_otime = start_time+params[params_set_counter].offset;
	//cout << "opening Target file " << file_name << " to write " << (array_length*sizeof(float)) << " bytes reset? " << reset_file << endl;
	auto err = fopen_s(&pfp,file_name, reset_file ? "wb" : "ab");
	//cout << "opening now tesing " << file_name << endl;
	if (err!=0) {
		PrintFormat("cant open Target file %s\n" ,file_name);
		throw std::runtime_error("cant open Target file");
	}
	//cout << "file " << file_name << " opened for writing "  << endl;
	//printf("file %s opened for writing\n",file_name);
	i=0;
	while( i < array_length) {
		if ( include_time ) {
			fwrite(&(t_otime),sizeof(float), 1, pfp);
		}
		
		/*
		if ( i == 0 ) {
			for(int j=0;j<chunk_size;j++){
				printf("buffer[%d]=%.3e\n",j,buffer[j]);
			}
		}
		*/
		fwrite((subject+i),sizeof(float),chunk_size,pfp);
		i+=chunk_size;
		t_otime += static_cast<float>(params[params_set_counter].Ts());
		//printf("written %d bytes, %.1f%% out of %d\n",i,(float)(100*i/array_length),array_length);
	}
	//cout << "file complete, closing " << file_name << endl;
	fclose(pfp);
}


void AudioGramCreator::saveFloatArrayToDisk(const std::vector<float>& subject,int array_length,int chunk_size,const char *file_name,bool include_time=true,bool reset_file=true) {
	int i;
	FILE *pfp;
	float t_otime = 0;
	//printf("opening BM file %s to write %d bytes reset? %d\n",file_name,array_length*sizeof(float),reset_file);
	fopen_s(&pfp, file_name, reset_file ? "wb" : "ab");
	if (!pfp) {
		PrintFormat("Can't open file %s\n", file_name);
		throw std::runtime_error("cant open save float array file");
	}
	//printf("file %s opened for writing\n",file_name);
	i=0;
	while( i < array_length) {
		if ( include_time ) {
			fwrite(&(t_otime),sizeof(float), 1, pfp);
		}
		/*
		if ( i == 0 ) {
			for(int j=0;j<chunk_size;j++){
				printf("buffer[%d]=%.3e\n",j,buffer[j]);
			}
		}
		*/
		fwrite(&subject[i],sizeof(float),chunk_size,pfp);
		i+=chunk_size;
		t_otime += static_cast<float>(params[params_set_counter].Ts());
		//printf("written %d bytes, %.1f%% out of %d\n",i,(float)(100*i/array_length),array_length);
	}
	//printf("file complete, closing %s\n",file_name);
	fclose(pfp);
}
void AudioGramCreator::loadFloatArrayFromDisk(float *subject,int array_length,const char *file_name) {
	int i;
	FILE *pfp;
	float t_otime = 0;
	//printf("opening BM file %s to write %d bytes reset? %d\n",file_name,array_length*sizeof(float),reset_file);
	fopen_s(&pfp, file_name, "rb");
	if (!pfp) {
		PrintFormat("Can't open file %s\n", file_name);
		throw std::runtime_error("cant open save float array file");
	}
	//printf("file %s opened for writing\n",file_name);
	i=0;
	fread(subject,sizeof(float),array_length,pfp);
	//printf("file complete, closing %s\n",file_name);
	fclose(pfp);
}
void AudioGramCreator::loadOutputArrayFromDisk(float *subject,int array_length,int chunk_size,const char *file_name,bool include_time=true) {
	int i;
	FILE *pfp;
	float t_otime = 0;
	//printf("opening BM file %s to write %d bytes reset? %d\n",file_name,array_length*sizeof(float),reset_file);
	fopen_s(&pfp, file_name, "rb");
	if (!pfp) {
		PrintFormat("Can't open file %s\n", file_name);
		throw std::runtime_error("cant open save float array file");
	}
	//printf("file %s opened for writing\n",file_name);
	i=0;
	while( i < array_length) {
		if ( include_time ) {
			fread(&(t_otime),sizeof(float), 1, pfp);
		}
		fread((subject+i),sizeof(float),chunk_size,pfp);
		i+=chunk_size;
	}
	printf("loaded %d data noddes from %s final time is %.2f\n",array_length,file_name,t_otime);
	fclose(pfp);
}

void AudioGramCreator::loadSavedSpeeds(std::vector<float>& input) {

	for(int i=0;i<AudioGramCreator::writeMatrixSize();i++) {
		//BM_input[i + timeIgnored] = input[i];
		BM_input[i] = input[i];
	}
}

void AudioGramCreator::ac_filter() {
	//printf("lambda calculating the filter is fir? %d, order of filter=%d\n",is_filter_fir,filter_size);
	if ( is_filter_fir ) {
		FIRFilterTemplate<float, lambdaFloat, double>(BM_input, AC, tffull.Numerator,sections, lambdaOffset(), time_dimension, time_blocks);
	} else { 
		IIRFilterTemplate<float, lambdaFloat, double>(BM_input, AC, tffull.Denominator, tffull.Numerator, sections, lambdaOffset(), time_dimension, time_blocks);
	}
	
}

void AudioGramCreator::IHCCalc() {
	double eta_AC = params[params_set_counter].eta_AC*params[params_set_counter].scaleBMVelocityForLambdaCalculation;
	double eta_DC = params[params_set_counter].eta_DC*params[params_set_counter].scaleBMVelocityForLambdaCalculation*params[params_set_counter].scaleBMVelocityForLambdaCalculation;
	IHC = (static_cast<lambdaFloat>(eta_AC)*AC) + (static_cast<lambdaFloat>(eta_DC)*DC);
	int sectionsbit = sections-1; // bits of sections number is 2^n -1 for bitwise remainder
	
	for(int n = 0; n<matrixSize(); n++) {
		int i = n&sectionsbit; //n/time_dimension; untransposed this the section pointer cycles instead of being single block,  bitwise and
		IHC[n] = max(IHC[n],static_cast<lambdaFloat>(0.0));
		lg10[n] = static_cast<lambdaFloat>(log10(IHC_damage_factor[i]*IHC[n]+EPS));
	}
	// this is actually like pre IHC
	
}

void AudioGramCreator::createDS() {
	vectorSumTemplate<float, lambdaFloat, lambdaFloat>(BM_input, 1.0f, AC, -1.0, Shigh);
	dS = Shigh*Shigh;
}

void AudioGramCreator::filterDC (
	) {
	AudioGramCreator::createDS();
	// dc filter values load on setup
	FIRFilterTemplate<lambdaFloat, lambdaFloat, double>(dS, DC, filter_dc, sections, lambdaOffset(), time_dimension, time_blocks);
	

}

void AudioGramCreator::calcLambda(int lambda_index) {
	float *l = Lambda.data() + lambda_index*matrixSize();
	//float *dl = dLambda.data() + lambda_index*matrixSize();
	//float *dsl = dSquareLambda.data() + lambda_index*matrixSize();
	float spo = params[params_set_counter].spontRate[lambda_index];
	//cout << "lambda sat: " << params[params_set_counter].Lambda_SAT << "\n";
	for(int n = 0; n<matrixSize(); n++) {
		l[n] = min(params[params_set_counter].Lambda_SAT, params[params_set_counter].spontRate[lambda_index] + std::max(static_cast<float>(params[params_set_counter].Aihc[lambda_index] * lg10[n]), 0.0f));
		//dl[n] = (l[n] - spo)/dA;
		//dsl[n] = dl[n]*dl[n];
    }
}
void AudioGramCreator::calcLambdas(Log &outer_log) {
	AudioGramCreator::IHCCalc();
	for(int i=0;i<LAMBDA_COUNT;i++){
		AudioGramCreator::calcLambda(i);
	}
	if (params[params_set_counter].Calculate_JND) {
		AudioGramCreator::calcJND(outer_log);
	}
	//cout << "lambda size calc " << (LAMBDA_COUNT*matrixSize()) << "\n";
}
void AudioGramCreator::saveLambdas(size_t overlap_reduce, size_t overlap_offset,Log &integrated_log) {
	//printf("saving lambdas started offset %d,length=%d\n",timeIgnoredMatrix(),writeMatrixSize());
	//int show_transient = params[params_set_counter].show_transient;
	//if (!is_first_time_for_set&&params[params_set_counter].Decouple_Filter==0) show_transient = 0;
	//int overlapNodesOffsetFix = (1 - show_transient)*params[params_set_counter].calcOverlapNodes()*SECTIONS;
	integrated_log.markTime(6);
	const std::vector<std::string> lambdas_names = vector<std::string>{"lambda_high", "lambda_medium", "lambda_low"};
	for (int i = 0; i < LAMBDA_COUNT; i++) {
		// + (is_first_time_for_set ? 0 : timeIgnoredMatrix())
		if (params[params_set_counter].Allowed_Output_Flags(i+1)) params[params_set_counter].vhout->write_vector(lambdas_names[i], Lambda, writeMatrixSize() - overlap_reduce, overlap_offset + i*matrixSize(), SECTIONS);
		//obuf.append_buffer(Lambda, writeMatrixSize() - overlap_reduce, overlap_offset + i*matrixSize());
		//cout << "appending lambda #" << i << " from offset " << (overlapNodesOffsetFix + i*matrixSize()) << ", for #" << (writeMatrixSize() - overlapNodesOffsetFix) << " nodes\n";
	}
	integrated_log.setMarkedValueAtIndex(50, static_cast<float>(12 * writeMatrixSize()), params[params_set_counter].Show_CPU_Run_Time & 8);
	integrated_log.elapsedTimeInterrupt(std::string("Save lambda To Memory size= ") + std::to_string(12 * writeMatrixSize()) + std::string("Bytes"), 6, 7, params[params_set_counter].Show_CPU_Run_Time & 8);
	if (params[params_set_counter].Show_CPU_Run_Time & 256) {
		integrated_log.markTime(11);
		AudioGramCreator::saveStageToDisk(Lambda.data(),static_cast<int>(matrixSize()), static_cast<int>(writeMatrixSize()), sections, params[params_set_counter].lambda_high_file_name, true, true);
		AudioGramCreator::saveStageToDisk(Lambda.data() + matrixSize(), static_cast<int>(matrixSize()), static_cast<int>(writeMatrixSize()), sections, params[params_set_counter].lambda_medium_file_name, true, true);
		AudioGramCreator::saveStageToDisk(Lambda.data() + 2 * matrixSize(), static_cast<int>(matrixSize()), static_cast<int>(writeMatrixSize()), sections, params[params_set_counter].lambda_low_file_name, true, true);
		
		integrated_log.elapsedTimeInterrupt(std::string("Save lambda To HD size= ")+std::to_string(12* writeMatrixSize())+std::string("Bytes"), 11, 12, params[params_set_counter].Show_CPU_Run_Time & 256);
	}
}
 // for untransposed coordinates will jump by sections
void AudioGramCreator::calcSum(float *src,float *dst,int cells,int jump_cells) {
	dst[0] = 0;
	for(int i=0;i<cells;i++) {
		dst[0] += src[i*jump_cells];
	}
}
 // for untransposed coordinates will jump by sections
void AudioGramCreator::calcAvg(float *src,float *dst,int cells,int jump_cells) {
	AudioGramCreator::calcSum(src,dst,cells,jump_cells);
	dst[0] = dst[0]/cells;
}

// also calcs nIHC
void AudioGramCreator::calcdA(std::vector<float>& input) {
	if (!input.empty()) {
		int relevantNodes = static_cast<int>(write_time_dimension);
		size_t overlapNodes = params[params_set_counter].calcOverlapNodes();
		int JNDIntervalLength = params[params_set_counter].JND_Interval_Nodes_Full();
		relevantNodes = relevantNodes - static_cast<int>(overlapNodes);
		int JNDIntervals = relevantNodes / JNDIntervalLength;
		dA = vector<double>(JNDIntervals, 0.0);
		int headOffset = params[params_set_counter].JND_Interval_Nodes_Offset();
		int lengthOffset = params[params_set_counter].JND_Interval_Nodes_Length();
		for (int dAindex = 0; dAindex < JNDIntervals; dAindex++) {
			int start_interval = dAindex*JNDIntervalLength + headOffset + static_cast<int>(overlapNodes);
			for (int input_index = start_interval; input_index < start_interval + lengthOffset; input_index++) {
				if (dA[dAindex] < input[input_index] * input[input_index]) {
					dA[dAindex] = input[input_index] * input[input_index];
				}
			}
			dA[dAindex] = sqrt(dA[dAindex]); // division happens since
		}
	}
}
void AudioGramCreator::calcJND(Log &outer_log) {
	size_t relevantNodes = write_time_dimension;
	//cout << "relevant nodes: " << relevantNodes << "\n";
	size_t overlapNodes = params[params_set_counter].calcOverlapNodes();
	int JNDIntervalLength = params[params_set_counter].JND_Interval_Nodes_Full();
	size_t lambdaFullSize = params[params_set_counter].run_ihc_on_cpu ? writeMatrixSize() : matrixSize();
	relevantNodes = relevantNodes - overlapNodes;
	
	bool calculate_rms = params[params_set_counter].JND_Calculate_RMS();
	bool calculate_ai = params[params_set_counter].JND_Calculate_AI();
	int JNDIntervals = static_cast<int>(relevantNodes) / JNDIntervalLength;
	size_t JNDAllIntervals = params[params_set_counter].numberOfJNDIntervals();
	int headOffset = params[params_set_counter].JND_Interval_Nodes_Offset();
	int lengthOffset = params[params_set_counter].JND_Interval_Nodes_Length();
	double Tmean = static_cast<double>(lengthOffset*params[params_set_counter].Ts());
	double Ts = static_cast<double>(params[params_set_counter].Ts());
	double dLambdaCalculated;
	double refLambda;
	double preFisherAIValue;
	if (!params[params_set_counter].Calculate_JND_On_CPU) {
		CudaCalculateJND(
			calculate_ai,
			calculate_rms,
			getMeanNodes(),
			getFisherNodes(),
			params[params_set_counter].SPLRefVal,
			params[params_set_counter].Fs,
			params[params_set_counter].scaleBMVelocityForLambdaCalculation,
			nIHC.data(),
			params[params_set_counter].JND_Calculated_Intervals_Positions.data(),
			static_cast<int>(params[params_set_counter].JND_Calculated_Intervals_Positions.size()),
			params[params_set_counter].JND_Reference_Intervals_Positions.data(),
			static_cast<int>(params[params_set_counter].JND_Reference_Intervals_Positions.size()),
			handeledIntervalsJND,
			static_cast<int>(params[params_set_counter].numberOfJNDIntervals()), // global number of JND intervals 
			JNDIntervals, // current input # of handeled intervals
			headOffset,
			static_cast<int>(overlapNodes),
			JNDIntervalLength,
			lengthOffset, // local not global
			params[params_set_counter].JND_Serial_Intervals_Positions.data(),
			params[params_set_counter].JND_Interval_To_Reference.data(),
			F_RA.data(), // result for fisher rate not lambda summed, but time and space reduced
			FisherAISum.data(), // result for fisher AI not lambda summed, but time and space reduced	
			static_cast<int>(writeMatrixSize()),
			static_cast<int>(matrixSize()),
			params[params_set_counter].JND_Delta_Alpha_Length_Factor(),
			params[params_set_counter].JND_USE_Spont_Base_Reference_Lambda,
			params[params_set_counter].Show_Run_Time,
			!params[params_set_counter].Generating_Input_Profile(), // if generating the input profile, no dA is needed
			(params[params_set_counter].Show_Generated_Input & 2)>0, 
			params[params_set_counter].calculateBackupStage(),
			outer_log);
	} else {
		PrintFormat("Calculating JND on CPU\n");
		if (calculate_rms) {
			for (int lambda_index = 0; lambda_index < LAMBDA_COUNT; lambda_index++) {	// summed lambda
				for (int dAindex = 0; dAindex < JNDIntervals; dAindex++) {	// calculated interval index
					int globaldAIndex = dAindex + handeledIntervalsJND;
					for (int section_index = 0; section_index < sections; section_index++) {
						/**
						* first stage is calculate mean rate (both reference and regular)
						* equivalent to matlab code: note marked lines
						* do note however that mean rate size is LAMBDA_COUNT*JNDIntervals*sections
						for k=1:3
						Bl=IHC2Lamda( IHC,A(k),spont(k));
						Lamda(k,sec,1:Time)=reshape(Bl,1,1,Time);
						--> Ml=mean(Bl);
						--> MeanRate(k,sec)=Ml;

						end
						*/
						int block_offset = lambda_index*static_cast<int>(lambdaFullSize) + section_index + sections*(dAindex*JNDIntervalLength + static_cast<int>(overlapNodes) + headOffset);
						
						int mean_rate_offset = (lambda_index*JNDIntervals + dAindex)*sections + section_index;
						double sumMean = 0.0;
						double midMean = 0.0;
						for (int i = 0; i < lengthOffset; i++) {
							midMean = Lambda[block_offset + i*sections] - params[params_set_counter].spontRate[lambda_index];
							sumMean += midMean;
						}
						sumMean = sumMean / static_cast<double>(lengthOffset);
						MeanRate[mean_rate_offset] = static_cast<JNDFloat>(sumMean);
					}
				}
			}
		}

		bool hasRefrences = !params[params_set_counter].JND_Reference_Intervals_Positions.empty();
		int test_x_times = 2;
		for (int lambda_index = 0; lambda_index < LAMBDA_COUNT; lambda_index++) {	// summed lambda
			for (int dAindex = 0; dAindex < JNDIntervals; dAindex++) {	// calculated interval index
				int globaldAIndex = dAindex + handeledIntervalsJND;
				int globalReferenceInterval = globaldAIndex;
				bool isReference = checkVector(params[params_set_counter].JND_Reference_Intervals_Positions, globaldAIndex);
				double dAvalue = params[params_set_counter].Generating_Input_Profile() ? params[params_set_counter].inputProfile[globaldAIndex].dA : dA[dAindex];
				if (!isReference&&checkVector(params[params_set_counter].JND_Calculated_Intervals_Positions, globalReferenceInterval)) {
					globalReferenceInterval = params[params_set_counter].JND_Interval_To_Reference[params[params_set_counter].JND_Serial_Intervals_Positions[globalReferenceInterval]];
				}
				int dAreferenceIndex = globalReferenceInterval - handeledIntervalsJND; // to find local index on tested output
				// globaldAIndex is fix for previous handeled indexes


				for (int section_index = 0; section_index < sections; section_index++) {
					/**
					* seconds stage is calculating delta mean rate and delta lambda as
					* dLamda=(TestLamda-RefLamda)./dA;
					* dMeanRate=(TestMeanRate-RefMeanRate)./dA;
					*
					* note, each refrence will be calculated according to dA
					* also references will be zeroed in this arrays
					*/

					// here I calculate the mean rate
					int mean_rate_calculate_offset = (lambda_index*JNDIntervals + dAindex)*sections + section_index;
					int mean_rate_reference_offset = (lambda_index*JNDIntervals + dAreferenceIndex)*sections + section_index;
					
					if (calculate_rms) {
						dMeanRate[mean_rate_calculate_offset] = dAvalue > 0 ? static_cast<JNDFloat>((MeanRate[mean_rate_calculate_offset] - MeanRate[mean_rate_reference_offset]) / dAvalue) : static_cast<JNDFloat>(0);
					}
					// for inner summary of fisher AI in line 
					// FisherAI(j)=W(j)*sum(nIHC.*sum(Ts*(dL.^2./reshape(RefLamda(j,:,:),Nsec,Time))'));
					if (calculate_ai) {
						double preFisherAITimeReducedValue = 0.0f;
						// here I calculate the lambda
						int calculate_lambda_offset = lambda_index*static_cast<int>(lambdaFullSize) + section_index + sections*(dAindex*JNDIntervalLength + static_cast<int>(overlapNodes) + headOffset);
						int reference_lambda_offset = lambda_index*static_cast<int>(lambdaFullSize) + section_index + sections*(dAreferenceIndex*JNDIntervalLength + static_cast<int>(overlapNodes) + headOffset);
						for (int time_offset = 0; time_offset < lengthOffset; time_offset++) {
							refLambda = Lambda[reference_lambda_offset];
							dLambdaCalculated = dAvalue > 0 ? (Lambda[calculate_lambda_offset] - refLambda) / dAvalue : 0;
							dLambda[calculate_lambda_offset] = static_cast<JNDFloat>(dLambdaCalculated);
							/*
							* calculating pre fisher AI
							* from matlab
							* fisher AI : Ts*(dL.^2./reshape(RefLamda(j,:,:),Nsec,Time)	=> into pre fisher AI
							*/
							preFisherAIValue = Ts*dLambdaCalculated*dLambdaCalculated / refLambda;
							preFisherAI[calculate_lambda_offset] = static_cast<JNDFloat>(preFisherAIValue);
							preFisherAITimeReducedValue += preFisherAIValue;
							reference_lambda_offset += sections;
							calculate_lambda_offset += sections;
						}
						preFisherAITimeReduced[mean_rate_calculate_offset] = static_cast<JNDFloat>(nIHC[section_index] * preFisherAITimeReducedValue);
					}
					/*
					* calculate pre fisher values before summering
					* from matlab
					* fisher rate : nIHC.*Tmean./RefMeanRate(j,:).*(dMeanRate(j,:).^2)	 => into pre fisher rate
					*/
					if (calculate_rms) {
						CRLB_RA[mean_rate_calculate_offset] = static_cast<JNDFloat>(nIHC[section_index] * Tmean) / (MeanRate[mean_rate_reference_offset] + params[params_set_counter].spontRate[lambda_index]) * (dMeanRate[mean_rate_calculate_offset] * dMeanRate[mean_rate_calculate_offset]);
					}

				}
			}
		}

		for (int dAindex = 0; dAindex < JNDIntervals; dAindex++) {	// calculated interval index
			int globaldAIndex = dAindex + handeledIntervalsJND;
			/**
			* fourth stage, outer summary and W multiplication in
			* FisherAI(j)=W(j)*sum(nIHC.*sum(Ts*(dL.^2./reshape(RefLamda(j,:,:),Nsec,Time))'));
			* and summary and W multiplication in
			* JND_RA(j)=W(j)*sum(nIHC.*Tmean./RefMeanRate(j,:).*(dMeanRate(j,:).^2));
			* this is basically summary on spatial dimension
			*/

			for (int lambda_index = 0; lambda_index < LAMBDA_COUNT; lambda_index++) {	// summed lambda

				double fisherAISummation = 0.0f;
				double fisherRateSummation = 0.0f;
				int mean_rate_calculate_offset = (lambda_index*JNDIntervals + dAindex)*sections;
				for (int section_index = 0; section_index < sections; section_index++) {
					if (calculate_rms) fisherRateSummation += CRLB_RA[mean_rate_calculate_offset];
					if (calculate_ai)fisherAISummation += preFisherAITimeReduced[mean_rate_calculate_offset];
					mean_rate_calculate_offset++;
				}
				if (calculate_rms) {
					fisherRateSummation = fisherRateSummation*params[params_set_counter].W[lambda_index];
					F_RA[lambda_index*JNDAllIntervals + globaldAIndex] = static_cast<JNDFloat>(fisherRateSummation);
				}
				if (calculate_ai) {
					fisherAISummation = fisherAISummation*params[params_set_counter].W[lambda_index];
					FisherAISum[lambda_index*JNDAllIntervals + globaldAIndex] = static_cast<JNDFloat>(fisherAISummation);
				}

			}
		}		
	}
	AudioGramCreator::calcJNDFinal();
	handeledIntervalsJND += JNDIntervals;
}

void AudioGramCreator::calcJNDFinal() {
	int relevantNodes = static_cast<int>(write_time_dimension);
	int overlapNodes = static_cast<int>(params[params_set_counter].calcOverlapNodes());
	int JNDIntervalLength = params[params_set_counter].JND_Interval_Nodes_Full();
	int JNDAllIntervals = static_cast<int>(params[params_set_counter].numberOfJNDIntervals());
	if (Failed_Converged_Signals.empty()) {
		Failed_Converged_Signals = std::vector<int>(JNDAllIntervals,0);
	}
	relevantNodes = relevantNodes - overlapNodes;
	bool calculate_rms = params[params_set_counter].JND_Calculate_RMS();
	bool calculate_ai = params[params_set_counter].JND_Calculate_AI();

	int JNDIntervals = relevantNodes / JNDIntervalLength;
	int positive_coupler = params[params_set_counter].Decouple_Filter > 0 ? params[params_set_counter].Decouple_Filter : static_cast<int>(Failed_Converged_Blocks.size());
	for (int dAindex = 0; dAindex < JNDIntervals; dAindex++) {	// calculated interval index
		int globaldAIndex = dAindex + handeledIntervalsJND;
		int globalReferenceInterval = globaldAIndex;
		bool isReference = checkVector(params[params_set_counter].JND_Reference_Intervals_Positions, globaldAIndex);
		if (!isReference&&checkVector(params[params_set_counter].JND_Calculated_Intervals_Positions, globalReferenceInterval)) {
			globalReferenceInterval = params[params_set_counter].JND_Interval_To_Reference[params[params_set_counter].JND_Serial_Intervals_Positions[globalReferenceInterval]];
		}
		int dAreferenceIndex = globalReferenceInterval - handeledIntervalsJND; // to find local index on tested output
		double fisherAIReg = 0.0;
		double fisherRateReg = 0.0;
		for (int lambda_index = 0; lambda_index < LAMBDA_COUNT; lambda_index++) {	// summed lambda
			/**
			* fifth stage
			* the summary
			* F_RA=sqrt(sum(JND_RA));
			* FisherAISum=sqrt(sum( FisherAI));
			*/
			fisherAIReg += FisherAISum[lambda_index*JNDAllIntervals + globaldAIndex];

			fisherRateReg += F_RA[lambda_index*JNDAllIntervals + globaldAIndex];
		}
		if (!isReference) {
			int result_interval_index = params[params_set_counter].JND_Serial_Intervals_Positions[globaldAIndex];
			for (int pind = 0; pind < positive_coupler; pind++) {
				Failed_Converged_Signals[result_interval_index] += Failed_Converged_Blocks[pind+ dAindex*positive_coupler];
			}
			if (calculate_ai) {
				double sfisherAIReg = 1.0/sqrt(fisherAIReg);
				FisherAI[result_interval_index] = static_cast<JNDFloat>(sfisherAIReg);
				AiJNDall[result_interval_index] = params[params_set_counter].PA2SPLForFinalJNDBounded(sfisherAIReg);// 20 * log10f(1 / sfisherAIReg / params[params_set_counter].SPLRefVal);
			}
			if (calculate_rms) {
				double sfisherRateReg = 1.0/sqrt(fisherRateReg);
				
				JND_RA[result_interval_index] = static_cast<JNDFloat>(sfisherRateReg);
				RateJNDall[result_interval_index] = params[params_set_counter].PA2SPLForFinalJNDBounded(sfisherRateReg);
			}
			
		}
	}
}


std::string AudioGramCreator::getSimpleLegend(const int& isRMS, const int& global_interval) {
	std::ostringstream oss("");
	oss.setf(oss.boolalpha);
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(0);
	size_t M = params[params_set_counter].JNDInputTypes();
	if (isRMS) {
		oss << "Rate";
	} else {
		oss << "AI";
	}
	oss << " ";
	oss << params[params_set_counter].inputProfile[global_interval].dBSPLSignal << " dB,";
	if (params[params_set_counter].inputProfile[global_interval].dBSPLNoise < MIN_INF_POWER_LEVEL && params[params_set_counter].inputProfile[global_interval].Wn > 0) {
		oss << "Noise " << params[params_set_counter].inputProfile[global_interval].dBSPLNoise << " dB";
	} else {
		oss << "Noiseless";
	}
	return oss.str();
}
std::vector<std::string> AudioGramCreator::getSimpleLegends(const int& isRMS, const std::vector<int>& intervals) {
	auto vs = std::vector<std::string>();
	std::vector<double> prevSPLPowers;
	size_t num_of_signals = params[params_set_counter].JNDInputTypes();
	size_t cycler = 0;
	for (auto interval : intervals) {
		auto profile = params[params_set_counter].inputProfile[interval];
		if (!profile.isReference /* && !checkVector(prevSPLPowers, profile.dBSPLSignal)*/ ) {
			if (num_of_signals == 1 || (cycler%num_of_signals) == 0) {
				vs.push_back(getSimpleLegend(isRMS, interval));
			}
			prevSPLPowers.push_back(profile.dBSPLSignal);
			cycler++;
		}
	}
	return vs;
}
void AudioGramCreator::calcComplexJND(const vector<double>& values) {
	if (params[params_set_counter].View_Complex_JND_Source_Values) {
		PrintFormat("raw values for jnd are: \n %s \n", viewVector<double>(values, static_cast<int>(params[params_set_counter].JNDInputTypes())).c_str());
	}
	
	int profiles_num = static_cast<int>(params[params_set_counter].complexProfiles.size());
	auto methods_flags = params[params_set_counter].JNDComplexMethods();
	for (int ind = 0; ind < profiles_num; ind++) {
		int method_index = 0;
		if (methods_flags.test(0)) {
			ApproximatedJNDall[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].calculateMinValue(values,Failed_Converged_Signals, params[params_set_counter].Approximated_JND_EPS_Diff, params[params_set_counter].View_Complex_JND_Source_Values);
			ApproximatedJNDallWarnings[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].calculateMinValueWarning;
			Failed_Converged_Summaries[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].Failed_Convergence_Warning;
			method_index++;

		}
		if (methods_flags.test(1)) {
			ApproximatedJNDall[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].calculateGradientMinMaxValue(values, Failed_Converged_Signals, params[params_set_counter].Approximated_JND_EPS_Diff, params[params_set_counter].View_Complex_JND_Source_Values);
			ApproximatedJNDallWarnings[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].calculateGradientMinMaxValueWarning;
			Failed_Converged_Summaries[ind + profiles_num*method_index] = params[params_set_counter].complexProfiles[ind].Failed_Convergence_Warning;
			method_index++;
		}
	}
}


std::vector<std::string> AudioGramCreator::getLegends(const int& is_complex) {
	auto vs = std::vector<std::string>();
	int profiles_num = static_cast<int>(params[params_set_counter].complexProfiles.size());
	auto methods_flags = params[params_set_counter].JNDComplexMethods();
	size_t M = params[params_set_counter].JNDInputTypes();
	int isRMS = params[params_set_counter].Calculate_From_Mean_Rate ? 4 : 2;
	int is_complex_one = (is_complex & 1) > 0 ? 1 : 0;
	int is_simple = (1-is_complex_one)* 8;
	for (int ind = 0; ind < profiles_num; ind++) {
		int method_index = 0;
		if (methods_flags.test(0)) {
			if (M == 1 || (ind%M) == 0 || is_complex == 0) vs.push_back(params[params_set_counter].complexProfiles[ind].showLegend(isRMS + is_simple +16));
			method_index++;

		}
		if (methods_flags.test(1)) {
			if (M == 1 || (ind%M) == 0 || is_complex == 0) vs.push_back(params[params_set_counter].complexProfiles[ind].showLegend(isRMS + is_simple  + 32));
			method_index++;
		}
	}
	return vs;
}
AudioGramCreator::AudioGramCreator(void) {

}
AudioGramCreator::AudioGramCreator(int t, int writeTime, int allocateTime, int sections_in_cochlea, CParams *tparams, CModel *model, double start_time, std::vector<float>& input)
{
	this->setupCreator(t,writeTime,allocateTime,sections_in_cochlea,tparams,model,start_time,true,0,input); // on first time allocating memory if lambda not disabled also params set is 0
}
void AudioGramCreator::valuesSetupCreator(int t, int writeTime, int x, CParams *tparams, CModel *input_model, double start_time_param, int params_set_counter, std::vector<float>& input)
{
	// model and parameters setup
	this->params_set_counter = params_set_counter;
	params = tparams;
	model= input_model;
	time_dimension = t;
	write_time_dimension = writeTime; // original size for writing at the end
	sections = x;
	start_time = (float)start_time_param;
	time_blocks = model->configurations[params_set_counter]._num_frequencies;
	if (is_first_time_for_set) {
		handeledIntervalsJND = 0;
	}
	// dA for each block will be avaraged across square value of input	and than sqrt
	if (params[params_set_counter].Calculate_JND && params[params_set_counter].Calculate_JND_On_CPU) {
		AudioGramCreator::calcdA(input);
	}
	
	DC_filter_size = static_cast<int>(round(params[params_set_counter].Fs*params[params_set_counter].Tdelta));
	
	
	// value loading subcritical to overall performance will be set anyway
	fisher_size = LAMBDA_COUNT*params[params_set_counter].numberOfJNDIntervals();
	setFisherNodes(static_cast<int>(fisher_size));
	mean_size = getFisherNodes()*sections;
	setMeanNodes(static_cast<int>(mean_size));
	loadAihc(params[params_set_counter].Aihc.data());
	for (int lambda_counter = 0; lambda_counter < LAMBDA_COUNT; lambda_counter++) {
		Nerves_Clusters[lambda_counter] = params[params_set_counter].spontRate[lambda_counter];
		//Nerves_Clusters[lambda_counter + LAMBDA_COUNT] = params[params_set_counter].Aihc[lambda_counter];
		Nerves_Clusters[lambda_counter + 2 * LAMBDA_COUNT] = params[params_set_counter].W[lambda_counter];
	}
}

void AudioGramCreator::setupCreator(int t, int writeTime, int allocateTime, int sections_in_cochlea, CParams *tparams, CModel *input_model, double start_time_param, bool first_time, int params_set_counter, std::vector<float>& input)
{
	audiogramlog.markTime(13);
	allocate_time = allocateTime;
	is_first_time_for_set = first_time;
	AudioGramCreator::valuesSetupCreator(t, writeTime, sections_in_cochlea, tparams, input_model, start_time_param, params_set_counter, input);
	if (params_set_counter==0) {
		// dc filter size buffer
		filter_dc = std::vector<double>();
	}
	audiogramlog.markTime(0);
	/**
	* test for either lambda first enabled or first IHC cpu enforced to allocate vectors, 
	* IHC allocation scheme will not be activated unless first lambda is enabled since its meaningless
	*/
	int matrix_size = (allocateTime*sections_in_cochlea); // array will be defined only one adding memory for continous run

	setLambdaNodes(LAMBDA_COUNT*matrix_size);
	setBMVelocityNodes(matrix_size);
	if (input_model->firstLambdaEnabled() == params_set_counter || (HAS_PARAM_SET(input_model->firstLambdaEnabled()) && input_model->firstForceCPUOnIHC() == params_set_counter)) {
		//cout << "first lambda enabled is " << params_set_counter << "\n";
		
		if (input_model->firstLambdaEnabled() == params_set_counter) {
			if (BM_input.empty()) {
				BM_input = std::vector<float>(getLambdaNodes(), 0.0f); // lambda count multiplications for debug purposes
				Lambda = std::vector<float>(getLambdaNodes(), 0.0f);
			}
			else if (BM_input.size() < getLambdaNodes()) {
				BM_input.resize(getLambdaNodes());
				Lambda.resize(getLambdaNodes());
			}

			// sections size buffers
			if (nIHC.empty()) {
				nIHC = std::vector<double>(sections, 0.0);
				IHC_damage_factor = std::vector<double>(sections, static_cast<double>(0.0));
				original_ihc = std::vector<float>(sections, 0.0);
				gamma = std::vector<float>(sections, 0.0f);
			}
		}
		if (input_model->firstForceCPUOnIHC() == params_set_counter) {
			// results size buffers
			if (AC.empty()) {
				AC = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);
				Shigh = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);
				dS = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);
				DC = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);

				IHC = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);
				lg10 = std::vector<lambdaFloat>(getBMVelocityNodes(), 0.0);
				buffer = std::vector<float>(getBMVelocityNodes(), 0.0f);
			}
			else if ( AC.size() < getBMVelocityNodes() ) {
				AC.resize(getBMVelocityNodes());
				Shigh.resize(getBMVelocityNodes());
				dS.resize(getBMVelocityNodes());
				DC.resize(getBMVelocityNodes());

				IHC.resize(getBMVelocityNodes());
				lg10.resize(getBMVelocityNodes());
				buffer.resize(getBMVelocityNodes());
			}
			
		}

		// lambda size buffers												
		if (cudaLambdaFloatShortBuffer.empty()) {
			dLambda = std::vector<JNDFloat>(getLambdaNodes(), 0.0);
			cudaLambdaFloatShortBuffer = std::vector<JNDFloat>(matrix_size, 0.0);
		}
		else if (cudaLambdaFloatShortBuffer.size() < getBMVelocityNodes()) {
			dLambda.resize(getLambdaNodes());
			cudaLambdaFloatShortBuffer.resize(getBMVelocityNodes());
		}
		if (HAS_PARAM_SET(input_model->firstJNDCalculationSetONCPU())) {
			if (cudaTestBuffer.empty()) {
				cudaTestBuffer = std::vector<float>(getLambdaNodes(), 0.0f);
				cudaLambdaFloatBuffer = std::vector<lambdaFloat>(getLambdaNodes(), 0.0);

				dSquareLambda = std::vector<JNDFloat>(getLambdaNodes(), 0.0);
				preFisherAI = std::vector<JNDFloat>(getLambdaNodes(), 0.0);
			}
			else if (cudaTestBuffer.size() < getLambdaNodes()) {
				cudaTestBuffer.resize(getLambdaNodes());
				cudaLambdaFloatBuffer.resize(getLambdaNodes());

				dSquareLambda.resize(getLambdaNodes());
				preFisherAI.resize(getLambdaNodes());

			}
		}



		// lambda size buffers

		if (CRLB_RA.empty()) {
			CRLB_RA = std::vector<JNDFloat>(getMeanNodes(), 0.0);
			preFisherAITimeReduced = std::vector<JNDFloat>(getMeanNodes(), 0.0);
		}
		else if (CRLB_RA.size() < getMeanNodes()) {
			CRLB_RA.resize(getMeanNodes());
			preFisherAITimeReduced.resize(getMeanNodes());
		}


		if (HAS_PARAM_SET(input_model->firstJNDCalculationSetONCPU())) {
			if (dSumLambda.empty()) {
				dSumLambda = std::vector<JNDFloat>(getMeanNodes(), 0.0);
				MeanRate = std::vector<JNDFloat>(getMeanNodes(), 0.0);
				dMeanRate = std::vector<JNDFloat>(getMeanNodes(), 0.0);
				dSquareMeanRate = std::vector<JNDFloat>(getMeanNodes(), 0.0);
			}
			else if (dSumLambda.size() < getMeanNodes()) {
				dSumLambda.resize(getMeanNodes());
				MeanRate.resize(getMeanNodes());
				dMeanRate.resize(getMeanNodes());
				dSquareMeanRate.resize(getMeanNodes());
			}
				
		}
		setFisherIntervals(static_cast<int>(params[params_set_counter].numberOfJNDIntervals()));
		setApproxJNDIntervals(params[params_set_counter].numberOfApproximatedJNDs());
		// fisher size buffers
		if (JND_RA.empty()) {
			JND_RA = std::vector<JNDFloat>(getFisherNodes(), 0.0);
			FisherAI = std::vector<JNDFloat>(getFisherNodes(), 0.0);
			F_RA = std::vector<JNDFloat>(getFisherNodes(), 0.0);
			FisherAISum = std::vector<JNDFloat>(getFisherNodes(), 0.0);
			AvgMeanRate = std::vector<JNDFloat>(getFisherNodes(), 0.0);
		} else if (JND_RA.size() < getFisherNodes() ) {
			JND_RA.resize(getFisherNodes());
			FisherAI.resize(getFisherNodes());
			F_RA.resize(getFisherNodes());
			FisherAISum.resize(getFisherNodes());
			AvgMeanRate.resize(getFisherNodes());
		}

		if (is_first_time_for_set) {
			// summed fisher size buffers	
			RateJNDall = std::vector<double>(getFisherIntervals(), 0.0);
			AiJNDall = std::vector<double>(getFisherIntervals(), 0.0);
			

			ApproximatedJNDall = std::vector<double>(getApproxJNDIntervals(), 0.0);
			ApproximatedJNDallWarnings = std::vector<int>(getApproxJNDIntervals(), 0);
			Failed_Converged_Summaries = std::vector<int>(getApproxJNDIntervals(), 0);
			if (params[params_set_counter].Show_JND_Configuration) {
				PrintFormat("sizeof(double) %d\n",sizeof(double));
				PrintFormat("sizeof(float) %d\n", sizeof(float));
				PrintFormat("mean_size = %d\n",getMeanNodes());
				PrintFormat("fisher_size = %d\n",getFisherNodes ());
				PrintFormat("fisher_intervals = %d\n",getFisherIntervals());
				if (params[params_set_counter].sim_type == SIM_TYPE_JND_COMPLEX_CALCULATIONS) {
					PrintFormat("approximated_jnd_intervals = %d => complex JND calculation is active\n", getApproxJNDIntervals());
				}
			}
		}

	}

	audiogramlog.markTime(1);
	nIHC = castVector<float, double>(params[params_set_counter].M_tot);
	// dc filter size buffer, resize anyway
	filter_dc.resize(DC_filter_size, 0.0);
	AudioGramCreator::setupIHCOHC();
	audiogramlog.markTime(2);
}
void AudioGramCreator::setupIHCOHC() {
	audiogramlog.markTime(3);
	if ( !params[params_set_counter].disable_lambda ) {
		
		for(int i=0;i<DC_filter_size;i++) {
			filter_dc[i] = static_cast<lambdaFloat>(1.0) / static_cast<lambdaFloat>(params[params_set_counter].Fs*params[params_set_counter].Tdelta);
		}
		//std::cout << "Dc filter, size = " << DC_filter_size << ",value = " << filter_dc[0] << std::endl;;
		AudioGramCreator::readFilter(model->configurations[params_set_counter]._ac_time_filter);
		for(int i=0;i<sections;i++){
			gamma[i] = static_cast<float>(model->configurations[params_set_counter]._gamma[i]);
			original_ihc[i] = static_cast<float>(model->configurations[params_set_counter]._nerves[i]);
			IHC_damage_factor[i] = pow(10.0,  double(original_ihc[i]));
			
		}
	}
	audiogramlog.markTime(4);
}

bool AudioGramCreator::isRunIncuda(void) {
	return (!params[params_set_counter].run_ihc_on_cpu);
}

void AudioGramCreator::runInCuda(float *host_bm_velocity,Log &outer_log) {
	if (params[params_set_counter].Review_AudioGramCall) {
		PrintFormat("write_time_dimension: %d\nallocate_time: %d\nTime blocks: %d\n", write_time_dimension, allocate_time, params[params_set_counter].Time_Blocks);
	}
	
	// run kernel initialization if lambda enabled and there is cuda run
	if (model->firstLambdaEnabled() <= params_set_counter && model->firstLambdaCUDARun() <= params_set_counter ) {
		audiogramlog.markTime(5);
		//throw std::runtime_error("Initiated failure pre IHC Kernel initialization");
		IHCNewKernel(
			IHC_damage_factor.data(),
			Nerves_Clusters,
			filter_dc.data(),
			DC_filter_size,
			tffull.Numerator.data(), // the ac 
			is_filter_fir ? tffull.Numerator.data() : tffull.Denominator.data(),
			!is_filter_fir,	// filter is IIR if its not FIR
			filter_size,
			sections,
			params[params_set_counter].Time_Blocks,
			params[params_set_counter].SPLRefVal,
			backup_speeds,
			0, ///params[params_set_counter].totalNodesLastBlockSaved(),
			static_cast<int>(time_dimension),
			static_cast<int>(write_time_dimension),
			static_cast<int>(allocate_time),
			static_cast<int>(params[params_set_counter].calcTimeBlockNodes()),
			0, //model->maxBackupNodesLength(),
			lambdaOffset(),
			params[params_set_counter].Lambda_SAT,
			params[params_set_counter].eta_AC,
			params[params_set_counter].eta_DC,
			params_set_counter >= model->firstLambdaEnabled() && params_set_counter == model->firstLambdaCUDARun() && is_first_time_for_set,
			is_first_time_for_set,
			false,//isLoadingBackupSpeedsFromDisk(),
			params[0].disable_advanced_memory_handling,
			params[params_set_counter].Review_Lambda_Memory,
			!isRunIncuda(),
			params[params_set_counter].scaleBMVelocityForLambdaCalculation,
			model->firstJNDCalculationSetONGPU()>=0,
			model->maxJNDIntervals(),
			static_cast<int>(params[params_set_counter].calcOverlapNodes()),
			params[params_set_counter].Decouple_Filter,
			params[params_set_counter].Show_Run_Time,
			outer_log);
		audiogramlog.markTime(6);
	}
	// since IHC will be always run on cuda, this tests if IHC calculation required
	if ( isRunIncuda() ) {
		audiogramlog.markTime(7);
		if (params[params_set_counter].Run_Stage_Before_Lambda()) {
			PrintFormat("Run Lambda on CUDA\n");
			// calculating the IHC kernel itself
			RunIHCKernel(cudaLambdaFloatShortBuffer.data(),params[params_set_counter].Show_Run_Time, params[params_set_counter].saveLambda(), params[params_set_counter].calculateBackupStage(), params[params_set_counter].Decouple_Unified_IHC_Factor,outer_log);
			// updates bm input for backup
			
			// JND Lambda spontanous fix
			
			if (params[params_set_counter].vh->hasVariable("TEST_File_Target")) {
				string filename_test = params[params_set_counter].vh->getValue<std::string >("TEST_File_Target");
				int backup_stage = params[params_set_counter].calculateBackupStage();
				if (backup_stage >= 9 && backup_stage <= 14) {
					//
					BM_input = castVector<JNDFloat, float>(cudaLambdaFloatShortBuffer);
				}
			}
		} else if (params[params_set_counter].Run_Stage_Lambda()) {
			PrintFormat("Upload Lambda to CUDA, calculated time nodes per neural profile = %d,written time nodes per neural profile = %d\n", time_dimension, write_time_dimension);
			for (int lambda_index = 0; lambda_index < LAMBDA_COUNT; lambda_index++) {
				params[params_set_counter].get_stage_data(dLambda, static_cast<int>(time_dimension)*lambda_index, static_cast<int>(write_time_dimension)*lambda_index, static_cast<int>(write_time_dimension), SECTIONS);
			}
			ReverseKernel_Copy_Results_Template<JNDFloat>(dLambda.data(), getCudaLambda(), 0, time_dimension*LAMBDA_COUNT, SECTIONS);
			updateCUDALambdaArray<JNDFloat>(getCudaLambda(), getCudaBuffer(), time_dimension, SECTIONS, params[params_set_counter].Show_Run_Time, params[params_set_counter].Show_Device_Data, params[params_set_counter].saveLambda(),outer_log);
		}
		audiogramlog.markTime(15);
		if (params[params_set_counter].saveLambda()) {
			BMOHCKernel_Copy_Lambda(dLambda.data(), time_dimension*SECTIONS*LAMBDA_COUNT, 0); // recovers lambda after calculation on cuda
			Lambda = castVector<JNDFloat, float>(dLambda);
		}
		audiogramlog.markTime(8);

		if (params[params_set_counter].Calculate_JND) {
			AudioGramCreator::calcJND(outer_log);
		}
		if (params[params_set_counter].vh->hasVariable("TEST_File_Target")) {
			string filename_test = params[params_set_counter].vh->getValue<std::string >("TEST_File_Target");
			int backup_stage = params[params_set_counter].calculateBackupStage();
			std::smatch sm;
			int target_size = getMeanNodes();
			if (backup_stage >= 9 && backup_stage <= 14) {

				target_size = static_cast<int>(writeMatrixSize());
			} else if (params[params_set_counter].Type_TEST_Output == "JND_Lambda") {
				target_size = static_cast<int>(LAMBDA_COUNT*matrixSize());
				GeneralKernel_Copy_Results_Template<JNDFloat>(dLambda.data(), getCudaLambda(), target_size);
				BM_input = castVector<JNDFloat, float>(dLambda);
			} else if (std::regex_match(params[params_set_counter].Type_TEST_Output, sm, std::regex("JND_Lambda([[:digit:]])"))) {
				target_size = static_cast<int>(matrixSize());
				int lambdaIndex = parseToScalar<int>(sm[1]);
				if (lambdaIndex >= LAMBDA_COUNT) {
					stringstream oss("");
					oss << "index of lambda must be smaller than " << LAMBDA_COUNT << ", you chose " << lambdaIndex;
					throw std::runtime_error(oss.str());
				}
				GeneralKernel_Copy_Results_Template<JNDFloat>(dLambda.data(), getCudaLambda(), target_size, lambdaIndex*target_size);
				BM_input = castVector<JNDFloat, float>(dLambda);
			}
			if (!params[params_set_counter].Calculate_JND_On_CPU) {
				
				if (params[params_set_counter].Type_TEST_Output == "dLambda") {
					target_size = static_cast<int>(LAMBDA_COUNT*matrixSize());
					GeneralKernel_Copy_Results_Template<JNDFloat>(dLambda.data(), getCudaBuffer(), target_size);
					BM_input = castVector<JNDFloat, float>(dLambda);
				} else if (std::regex_match(params[params_set_counter].Type_TEST_Output, sm, std::regex("dLambda([[:digit:]])"))) {
					target_size = static_cast<int>(matrixSize());
					int lambdaIndex = parseToScalar<int>(sm[1]);
					if (lambdaIndex >= LAMBDA_COUNT) {
						stringstream oss("");
						oss << "index of lambda must be smaller than " << LAMBDA_COUNT << ", you chose " << lambdaIndex;
						throw std::runtime_error(oss.str());
					}
					GeneralKernel_Copy_Results_Template<JNDFloat>(dLambda.data(), getCudaBuffer(), target_size, lambdaIndex*target_size);
					BM_input = castVector<JNDFloat, float>(dLambda);
				} else if (params[params_set_counter].Type_TEST_Output == "PRE_IHC") {
					GeneralKernel_Copy_Results_Template<lambdaFloat>(cudaLambdaFloatShortBuffer.data(), getCudaBuffer(), writeMatrixSize());
					BM_input = castVector<lambdaFloat, float>(cudaLambdaFloatShortBuffer);
					target_size = static_cast<int>(writeMatrixSize());
				} else if (params[params_set_counter].Type_TEST_Output == "dMeanRate") {
					GeneralKernel_Copy_Results_Template<JNDFloat>(CRLB_RA.data(), getCudaBuffer(), getMeanNodes());
					BM_input = castVector<JNDFloat, float>(CRLB_RA);
				} else if (params[params_set_counter].Type_TEST_Output == "MeanRate") {
					GeneralKernel_Copy_Results_Template<JNDFloat>(CRLB_RA.data(), getCudaMeanRate(), getMeanNodes());
					BM_input = castVector<JNDFloat, float>(CRLB_RA);
				} else if (params[params_set_counter].Type_TEST_Output == "CRLB_RA") {
					GeneralKernel_Copy_Results_Template<JNDFloat>(CRLB_RA.data(), getCudaBuffer(), getMeanNodes());
					BM_input = castVector<JNDFloat, float>(CRLB_RA);
				}
				saveArrayToDisk(BM_input.data(), target_size, sections, filename_test.c_str(), false, is_first_time_for_set); // backup BM speed		
			} else {
				BM_input = castVector<JNDFloat, float>(CRLB_RA);
				saveArrayToDisk(BM_input.data(), getMeanNodes(), sections, filename_test.c_str(), false, is_first_time_for_set); // backup BM speed
			}
		}
		
	}
}
void AudioGramCreator::freeAll(void) {
	audiogramlog.markTime(9);
	//if (HAS_PARAM_SET(model->firstLambdaEnabled()) && HAS_PARAM_SET(model->firstLambdaCUDARun())) {
		IHCKernel_Free();
	//}
	audiogramlog.markTime(10);
	audiogramlog.elapsedTimeView("Init Lambda Calculations values", 13, 0, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Init Lambda Calculations memory", 0, 1, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Load physical parameters", 1, 2, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("copy IHC/OHC", 3, 4, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Init IHC Kernel", 5, 6, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Run and copy IHC Kernel", 7, 8, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Post Processing Lambda", 15, 8, params[params_set_counter].Show_CPU_Run_Time & 2);
	audiogramlog.elapsedTimeView("Store Lambda array", 11, 12, params[params_set_counter].Show_CPU_Run_Time & 8);
	audiogramlog.elapsedTimeView("Free Lambda calculator", 9, 10, params[params_set_counter].Show_CPU_Run_Time & 4);
	//if ((params[params_set_counter].IS_MEX & 1) == 0) {
		audiogramlog.flushLog();
	//} else {
	//	audiogramlog.flushToIOHandler(params[params_set_counter].vhout, "audiogram_log");
	//}
}
AudioGramCreator::~AudioGramCreator(void)
{
	//cout << "AudioGramCreator destroyed...\n";
	AudioGramCreator::freeAll();
}
