#include "SubModel.h"



SubModel::SubModel(const SubModel &src) :
params(src.params),
_sections(src._sections),
_ac_time_filter(src._ac_time_filter),
_noise_filter(src._noise_filter),
_gamma(src._gamma),
_nerves(src._nerves),
_dbA(src._dbA),
_num_frequencies(src._num_frequencies)
{
	
}

SubModel::SubModel(CParams *input_Params, int sections) :
params(input_Params),
_sections(sections),
_ac_time_filter(5),
_noise_filter(5),
_gamma(sections),
_nerves(sections)
{
	_dbA = params->sin_dB;
	_num_frequencies = params->num_frequencies;
	//std::cout << "Starts gamma parameters set #" << i << "\n";
	char input_filename[FILE_NAME_LENGTH_MAX];

	// Init the OHC gamma vector
	_gamma = castVector<float,double>(params->ohc_vector);
	//std::cout << "_gamma vector: " << viewVector(_gamma,16) << std::endl;
	//std::cout << "ohc vector: " << viewVector(params->ohc_vector, 16) << std::endl;
	if (params->gamma_file_flag && params->ohc_mode == 0) {
		strcpy_s<FILE_NAME_LENGTH_MAX>(input_filename, params->gamma_file_name);
		CBin gamma_file(input_filename, BIN_READ);			// link to a gamma binary file.
		if (!gamma_file.read(_gamma))							// read the gamma vector into _gamma.
		{
			std::cout<< "ERROR - Unable to read <gamma> from file " << input_filename <<"\n";
			throw std::runtime_error("Unable to read <gamma> from file - 19");
			//MyError io_err("ERROR - Unable to read <gamma> from the file.", "Model.cpp");
			//cerr << io_err.what();
			//throw io_err;
		}
		gamma_file.close_file();
	}

	//std::cout << "Starts nerves analyzing of parameters set #" << i << "\n";
	// Init the IHC gamma vector
	_nerves = castVector<float, double>(params->ihc_vector);// Close the gamma file. 
	//std::cout << "ihc vector: " << viewVector(params->ihc_vector,16) << std::endl;
	//std::cout << "_nerves vector: " << viewVector(_nerves,16) << std::endl;
	if (params->nerve_file_flag && !params->disable_lambda && params->ihc_mode == 0 ) {
		strcpy_s<MAX_BUF_LENGTH>(input_filename, params->nerve_file_name);
		CBin nerves_file(input_filename, BIN_READ);			// link to a nerves binary file.
		if (!nerves_file.read(_nerves))							// read the nerves vector into _nerves.
		{
			std::cout << "ERROR - Unable to read <nerves> from file " << input_filename << "\n";
			throw std::runtime_error("Unable to read <nerves> from file - 19");
			//MyError io_err("ERROR - Unable to read <nerves> from the file.", "Model.cpp");
			//cerr << io_err.what();
			//throw io_err;
		}
		nerves_file.close_file();
	}
	if (params->Filter_Noise_Flag >0) {
		TBin<double> filterReader(params->Filter_Noise_File, BIN_READ);
		vector<double> filterBuffer(static_cast<unsigned int>(filterReader._file_length), 0);
		if (!filterReader.read(filterBuffer)) {
			stringstream oss("");
			oss << "ERROR - Unable to read <NoiseFilter> from file " << params->Filter_Noise_File;
			throw std::runtime_error(oss.str());
		}
		filterReader.close_file();
		_noise_filter.decodeBinFile(filterBuffer);

		if (params->Filter_Noise_Flag == 2) {
			_noise_filter.view(params->Fs);
		}
	}
	//std::cout << "Starts filter parameters \n";
	if (!params->disable_lambda) {
		if (params->ac_filter_mode == 1) {
			//std::cout << "functional filter parameters set analyzeACFilter\n";
			analyzeACFilter();
		}
		else {
			if (params->ac_filter_mode == 2) {
				_ac_time_filter.decodeBinFile(params->AC_Filter_Vector);
			}
			else {
				// filter mode 0 its a file

				strcpy_s<MAX_BUF_LENGTH>(input_filename, params->ac_filter_file_name);

				//std::cout << "file filter parameters set open,file: " << input_filename << "\n";
				CBin ac_time_filter_file(input_filename, BIN_READ);
				//std::cout << "file filter parameters bin loaded\n";
				vector<double> buffer(static_cast<unsigned int>(ac_time_filter_file._file_length), 0);
				//std::cout << "file filter parameters buffer defined\n";
				//_ac_time_filter[i].assign(boost::lexical_cast<unsigned int>(ac_time_filter_file._file_length), 0);
				if (!ac_time_filter_file.read(buffer)) {
					std::cout << "ERROR - Unable to read <time_filter> from file " << input_filename << "\n";
					throw std::runtime_error("Unable to read <time_filter> from file - 19");
				}
				//std::cout << "file filter parameters buffer read\n";
				ac_time_filter_file.close_file();
				//std::cout << "file filter parameters file closed\n";
				_ac_time_filter.decodeBinFile(buffer);
			}
			//std::cout << "file filter parameters file decoded\n";
			if (params->show_filter) {

				std::cout << "Filter_Source = " << input_filename << "\nOrder_Of_Filter = " << (_ac_time_filter.Numerator.size() - 1) << "\nFilter Structure=Direct Form I\nUnits=normalized(0 to 1)\n";
				_ac_time_filter.view(params->Fs);
			}
			
			
		}
	}
	//std::cout << "Finished sub model analyzing of parameters \n";
}


SubModel::~SubModel()
{
}


double SubModel::toLinear(double input){
	return pow(10, input / 20);
}
double SubModel::toDB(double input){
	return 20 * log10(input);
}
double SubModel::toDelta(double input){
	return toLinear(-1 * input);
}


/**
* using Oppenheim & Schafer 2nd addition digital signal processing book equation 7.104 at page 	502
* adding fixing factor for order number due to 20DB to dec
*/
int SubModel::calcMinimumOrder(double delta1, double delta2, double transitionWidth) {
	return static_cast<int>(ceil((-10 * log10(delta1*delta2) - 13) / (2.324*transitionWidth*M_PI))); 
}
PMOutput SubModel::calcFIRPM(const double& OmegaC, const double& transitionWidth, const double& weightPass, const double& weightStop, const size_t& NumTaps) {
	return firpm(NumTaps, { 0.0, OmegaC, OmegaC + transitionWidth, 1.0 }, { 1.0, 1.0, 0.0, 0.0 }, { weightPass, weightStop });
}

void SubModel::analyzeACFilter() {
	//	double FIRCoeff[MAX_NUMTAPS];        // FIR filter coefficients.  MAX_NUMTAPS = 256

	double OmegaC;                 // 0.0 < OmegaC < 1.0
	double BW = 0;                     // its lpf dont care about BW
	int NumTaps = params->FilterOrder; // filter order for calculating
	//double WinBeta;                // 0 <= WinBeta <= 10.0  This controls the Kaiser and Sinc windows.
	double ParksWidth;            // 0.01 <= ParksWidth <= 0.3 The transition bandwidth.
	TFIRPassTypes PassType = firLPF;     // firLPF, firHPF, firBPF, firNOTCH, firALLPASS  See FIRFilterCode.h
	//TWindowType WindowType;   // wtNONE, wtKAISER, wtSINC, and others.   See the FFT header file.
	//std::cout << "functional filter name " << params->filterName << "\n";
	string filterName = converToLower(params->filterName);
	//std::cout << "functional filter lower name " << filterName << "\n";
	int Fs = static_cast<int>(params->Fs);
	double NyQuistFrequency = static_cast<double>(params->Fs / 2);
	if (filterName == "equiripple") {
		// normalizing  fcut if its equirriple
		if (params->FilterOrder == -1) params->minOrder = true;
		// determine the omega c relevant to the parameters
		if (params->filtersMap.count("fc") == 0 && (params->filtersMap.count("fpass") == 0 || params->filtersMap.count("fstop") == 0)) {
			string strErr = "Filter ";
			strErr.append(params->filtersKeysStat[filterName]).append(" either Fpass,Fstop pair or Fc(as cut frequency) to properly calculate the filter");
			throw std::runtime_error(strErr);
		}
		double PW = 0;
		double maxFstop;
		double minFstop;
		bool hasBoundaries = params->filtersMap.count("fstop") > 0;
		if (params->filtersMap.count("fc")) {
			// Omega Cut calculated directly from Fcut
			OmegaC = static_cast<double>(params->Fc) / NyQuistFrequency;
		} else {
			// omega cut will be calculated from Fpass and fstop
			OmegaC = static_cast<double>(params->Fpass) / NyQuistFrequency;
			PW = static_cast<double>(params->Fstop - params->Fpass) / NyQuistFrequency;
			maxFstop = static_cast<double>(params->Fpass) + (NyQuistFrequency* 0.3);
			minFstop = static_cast<double>(params->Fpass) + (NyQuistFrequency* 0.01);
		}
		
		ParksWidth = PW;
		//std::cout << "functional TW boundaries " << minFstop << "to " << maxFstop << "\n";
		double deltaPass;
		double deltaStop;
		bool deltasCalculated = true; // true if deltas known
		if (params->filtersMap.count("order")) {
			NumTaps = params->FilterOrder;
		}
		if ( params->filtersMap.count("wstop") ) {
			if (!params->filtersMap.count("wpass"))  params->Wpass = 10;
			deltaPass = params->Wpass;
			deltaStop = params->Wstop;
		} else if (params->filtersMap.count("astop")) {
			// now will calculate deltas off Astop, Apass function
			if (!params->filtersMap.count("apass"))  params->Apass = 1;
			deltaPass = toDelta(params->Apass);
			deltaStop = toDelta(params->Astop);
		} else if (params->minOrder) {
			throw std::runtime_error("Equiripple filter must have either Wstop or Astop to calculate required minimum Order, alternatively specify filter order");
		} else {
			
			deltasCalculated = false;
			deltaPass = 1;
			deltaStop = 10;
		}


		// target stop ripple set but fpass and fstop not set so we need to search proper PW to ensure fstop
		bool search_pw = params->filtersMap.count("fc") > 0 && params->filtersMap.count("astop") > 0;
		if (search_pw) ParksWidth = 0.15;
		//std::cout << "functional delatas " << deltaPass << " to " << deltaStop << "\n";
		if (params->minOrder) {
			NumTaps = calcMinimumOrder(deltaPass, deltaStop, ParksWidth);
		}
		//std::cout << "Num Taps: " << NumTaps << "\n";
		//std::cout << "functional NumTaps " << NumTaps << "\n";

		double weightStop = 10;	// and weight stop as 10
		
		double weightPass = __tmax(0.01,10 * deltaStop / deltaPass);	// will normalized weight to 1 
		
		if (weightPass > 10 * deltaStop / deltaPass) {
			// weight pass is too high
			weightStop = weightStop*(weightPass / (10 * deltaStop / deltaPass));
		}
		if (params->filtersMap.count("wstop")) {
			weightStop = params->Wstop;
		}
		if (params->filtersMap.count("wpass")) {
			weightPass = params->Wpass;
		}
		//std::cout << "functional weights " << weightPass << " to " << weightStop << "\n";

		// transition bandwidth too large relevant only in case of fpass and fstop
		if (params->filtersMap.count("fc") == 0) {
			if (params->Fstop > maxFstop || params->Fstop < minFstop) {
				std::cout << "Fstop should be between " << minFstop << " and " << maxFstop
					<< " but is " << params->Fstop << "\n" << "Equirriple may not converge properly, aborting...\n";
				throw std::runtime_error("Unable to create viable equiripple - 1");
			}
		}
		/*
		NewParksMcClellan(FIRCoeff, NumTaps, PassType, OmegaC, 0, ParksWidth, weightPass, weightStop);

		// since McClellan converge at 0dB, fix is only needed if its larger than 0

		if (params->deltasFound) {
		// frequency correction
		FIRFreqError(FIRCoeff, NumTaps, PassType, &OmegaC, &BW, params->Apass);
		// calculate with normalized Omega
		NewParksMcClellan(FIRCoeff, NumTaps, PassType, OmegaC, 0, ParksWidth, weightPass, weightStop);
		}	 */
		
		//std::cout << "weightPass: " << weightPass << "weightStop: " << weightStop << "ParksWidth: " << ParksWidth << "OmegaC: " << OmegaC << "\n";
		PMOutput pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
		fixDCGain(pmo, 1);
		//std::cout << "2.weightPass: " << weightPass << "weightStop: " << weightStop << "ParksWidth: " << ParksWidth << "OmegaC: " << OmegaC << "\n";
		double OrigOmegaC = OmegaC;
		if (params->deltasFound) {
			// all important parameters are known, no degrees of freedom
			// frequency correction
			FIRFreqError(pmo, PassType, &OmegaC, params->filtersMap.count("apass")>0?params->Apass:3);
			//std::cout << "weightPass: " << weightPass << "weightStop: " << weightStop << "ParksWidth: " << ParksWidth << "OmegaC: " << OmegaC << "\n";
			// calculate with normalized omega
			pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
			fixDCGain(pmo, 1);
		}
		if (search_pw ) {
			//std::cout << "pmo delta is " << pmo.Q << "\n";
			int tests_done = 0;
			double prev_park_width_high = MAX_PARK_WIDTH;
			double prev_park_width_low = MIN_PARK_WIDTH;
			double fixFactor = 10;
			double prevQ = pmo.Q;
			while ((pmo.Q > 3 * deltaStop || 3 * pmo.Q < deltaStop) && tests_done < 10) {
				if (params->minOrder) {
					if ( pmo.Q > 10 * deltaStop || 10 * pmo.Q < deltaStop ) {
						NumTaps = NumTaps + static_cast<int>(ceil(10 * log10(pmo.Q / deltaStop)));
						if ((prevQ > deltaStop && pmo.Q < deltaStop) || (prevQ < deltaStop && pmo.Q > deltaStop)) {
							fixFactor = 0.5*fixFactor; // for better approach
						}
						prevQ = pmo.Q;
					} else {
						//  modify PW according to standard search methods
						if (pmo.Q > 3 * deltaStop) {
						  // delta too large meaning astop too small, increase PW
							prev_park_width_low = ParksWidth;
							ParksWidth = (ParksWidth + prev_park_width_high) / 2;
						} else {
							// delta too small pw is too large  , will search by taking medium between current PW and low PW not searched
							prev_park_width_high = ParksWidth;
							ParksWidth = (ParksWidth + prev_park_width_low) / 2;
						}
					}
				} else {
					if (pmo.Q > 3 * deltaStop) {
						// delta too large pw is not large enough , will search by taking medium between current PW and larger PW not searched
						prev_park_width_low = ParksWidth;
						ParksWidth = (ParksWidth + prev_park_width_high) / 2;
					}  else {
						// delta too small pw is too large  , will search by taking medium between current PW and low PW not searched
						prev_park_width_high = ParksWidth;
						ParksWidth = (ParksWidth + prev_park_width_low) / 2;
					}
				}
				OmegaC = OrigOmegaC; // lock omega c;

				pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
				fixDCGain(pmo, 1);
				 // fix omega only once to prevent repeating fixes
				FIRFreqError(pmo, PassType, &OmegaC, params->Apass);
				// calculate with normalized omega
				pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
				fixDCGain(pmo, 1);
				tests_done++;
				//std::cout << "pmo deltaStop DB is " << abs(toDB(pmo.Q)) << " and park width is " << ParksWidth << " wp=" << weightPass << " ws=" << weightStop << " at test #" << tests_done << "\n";
			}
		} else if (hasBoundaries && params->filtersMap.count("wstop") == 0 ) {
			int tests_done = 0;
			double fixFactor = 10;
			double prevQ = pmo.Q;
			while ( (pmo.Q > 3 * deltaStop || 3 * pmo.Q < deltaStop) && tests_done < 5) {
				if ((prevQ > deltaStop && pmo.Q < deltaStop) || (prevQ < deltaStop && pmo.Q > deltaStop)) {
					fixFactor = 0.5*fixFactor; // for better approach
				}
				prevQ = pmo.Q;
				NumTaps = NumTaps + static_cast<int>(ceil(fixFactor * log10(pmo.Q / deltaStop)));
				OmegaC = OrigOmegaC; // lock omega c;

				pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
				fixDCGain(pmo, 1);
				// fix omega only once to prevent repeating fixes
				FIRFreqError(pmo, PassType, &OmegaC, params->Apass);
				// calculate with normalized omega
				pmo = calcFIRPM(OmegaC, ParksWidth, weightPass, weightStop, NumTaps);
				fixDCGain(pmo, 1);
				tests_done++;
				//std::cout << "pmo deltaStop DB is " << abs(toDB(pmo.Q)) << " and park width is " << ParksWidth << " wp=" << weightPass << " ws=" << weightStop << " at test #" << tests_done << "\n";
			}
		}
		//hfunction = createFIRFunction(FIRCoeff, NumTaps);		// currently replace with firPM library and testing
		//std::cout << "loading filter\n";
		_ac_time_filter.load(createFIRFunction(pmo));
		//std::cout << "loadded filter\n";
		_ac_time_filter.setFIRGain(0); // normalize the filter
		//std::cout << "gain set filter\n";
		if (params->show_filter) {
			std::cout << "final test error in DB =" << toDB(pmo.delta) << "\ndeltaStop=" << abs(toDB(pmo.Q)) << "\niterations=" << pmo.iter << "\n";
			std::cout << "Filter_Type = " << params->filterName << "\nOrder_Of_Filter = " << NumTaps << "\nWeights stop= " << weightStop << "\nWeights pass= " << weightPass << "\nFilter Structure=Direct Form I\nUnits=normalized(0 to 1)\n";
			_ac_time_filter.view(Fs);
		}


	} else if (filterName == "window") {
		string windowNameType = params->filtersMap["windowtype"];
		TWindowType windowType = wtNONE;
		if (windowNameType == "kaiser") {
			windowType = wtKAISER;
		} else if (windowNameType == "kaiserbessel") {
			windowType = wtKAISER_BESSEL;
		} else if (windowNameType == "sinc") {
			windowType = wtSINC;
		} else if (windowNameType == "hanning") {
			windowType = wtHANNING;
		} else if (windowNameType == "hamming") {
			windowType = wtHAMMING;
		} else if (windowNameType == "blackmann") {
			windowType = wtBLACKMAN;
		} else if (windowNameType == "flattop") {
			windowType = wtFLATTOP;
		}
		if (params->filtersMap.count("order") > 0) {
			NumTaps = params->FilterOrder; 
		}
		double OmegaC;
		double Apass = params->filtersMap.count("apass") > 0 ? params->Apass : 3;
		double deltaPass = toDelta(Apass);
		double Astop = params->filtersMap.count("astop") > 0 ? params->Astop : 0;
		double deltaStop;
		if (Astop > 0) {
			deltaStop = toDelta(Astop);
		}
		double TW = 0; // transion width normalized to 1
		double beta = 0; // for kaiser
		if (params->filtersMap.count("fc") > 0) {
			OmegaC = static_cast<double>(params->Fc) / NyQuistFrequency;
			
		} else {
			// omega cut in the middle between fpass and fstop ideally
			OmegaC = static_cast<double>(params->Fpass+params->Fstop) / (2*NyQuistFrequency);
			TW = (params->Fstop - params->Fpass) / NyQuistFrequency;
			deltaStop = toDelta(Astop);
			// now the minimum order can be calculated from main lobe equation
			
			switch (windowType) {
				case wtNONE:
					NumTaps = static_cast<int>(4 * M_PI / TW);
					break;
				case wtHAMMING:
				case wtHANNING:
					NumTaps = static_cast<int>(8 * M_PI / TW);
					break;
				case wtBLACKMAN:
					NumTaps = static_cast<int>(12 * M_PI / TW);
					break;
				case wtBLACKMAN_HARRIS:
					NumTaps = static_cast<int>(9 * M_PI / TW);
					break;
				case wtKAISER:
					if (Astop == 0 && params->filtersMap.count("apass") == 0) {
						throw std::runtime_error("Kaiser window must have Astop or Apass for calculations");
					}
					double TWP = TW * M_PI;
					double maxA = __tmax(Apass, Astop);
					NumTaps = static_cast<int>(ceil(maxA > 21?((maxA - 7.95) / (2.285*TWP)) : (5.79/TWP)));
					if (params->filtersMap.count("beta") > 0) {
						beta = parseToScalar<double>(params->filtersMap["beta"]);
					} else if (maxA > 50) {
						beta = 0.1102*(maxA - 8.7);
					} else if (maxA > 21) {
						beta = 0.5842*pow(maxA - 21, 4) + 0.07886*(maxA - 21);
					}
					break;
			}
		}
		if (params->filtersMap.count("order") > 0) {
			NumTaps = params->FilterOrder;
		}
		vector<double> FIRCoeff = vector<double>(NumTaps, 0);
		RectWinFIR(FIRCoeff.data(), NumTaps, firLPF, OmegaC, 0);
		if (windowType != wtNONE) {
			FIRFilterWindow(FIRCoeff.data(), NumTaps, windowType, beta); // Use a window with RectWinFIR.
		}
		fixDCGain(FIRCoeff, 1);
		// fix omega only once to prevent repeating fixes
		FIRFreqError(FIRCoeff, PassType, &OmegaC,3);
		// calculate with normalized omega
		RectWinFIR(FIRCoeff.data(), NumTaps, firLPF, OmegaC, 0);
		if (windowType != wtNONE) {
			FIRFilterWindow(FIRCoeff.data(), NumTaps, windowType, beta); // Use a window with RectWinFIR.
		}
		fixDCGain(FIRCoeff, 1);

		_ac_time_filter = HFunction(FIRCoeff, vector<double>(1, 1));
		_ac_time_filter.setFIRGain(0); // normalize the filter
		
		if (params->show_filter) {
			
			std::cout << "Filter_Type = " << params->filterName << "(" << params->filtersMapRaw[params->filtersKeysStat["windowtype"]] << ")\nOrder_Of_Filter = " << NumTaps << "\nFilter Structure=Direct Form I\nUnits=normalized(0 to 1)\n";
			_ac_time_filter.view(Fs);
		}
	} else if (filterName == "butterworth") {
		double deltaPass;
		double deltaStop;
		deltaPass = toDelta(params->Apass);	 // power at pass band
		double H0 = toLinear(params->Apass);
		deltaStop = toDelta(params->Astop);
		double GainStop = toLinear(params->Astop);
		int filterOrder; // will be calculated from filter parameters

		// equations took from http://www.electronics-tutorials.ws/filter/filter_8.html
		double OmegaPass = static_cast<double>(params->Fpass) / NyQuistFrequency;
		double OmegaStop = static_cast<double>(params->Fstop) / NyQuistFrequency;
		double epsilon = sqrt(H0*H0 - 1);
		double doufilterOrder = log(sqrt(GainStop*GainStop - 1) / epsilon) / log(OmegaStop / OmegaPass);
		filterOrder = static_cast<int>(ceil(doufilterOrder));
		// calculate adjusted omega c from formula OmegaC^filterOrder = OmegaPass^filterOrder/epsilon	=> OmegaC = OmegaPass * epsilon^(-1/filterOrder)
		if (params->butterType == PASSBAND) {
			OmegaC = OmegaPass*pow(epsilon, -1.0 / static_cast<double>(filterOrder));
		}
		else {
			OmegaC = OmegaStop*pow(epsilon / sqrt(GainStop*GainStop - 1), 1.0 / static_cast<double>(filterOrder)); // this is good approx of stopband, need to add mode to seperate between passband and stopband
		}
		TIIRFilterParams iirParams;
		iirParams.IIRPassType = iirLPF;
		iirParams.ProtoType = BUTTERWORTH;
		iirParams.OmegaC = OmegaC;
		iirParams.NumPoles = filterOrder;
		iirParams.dBGain = 0;
		// get coefficents as biquads 
		//std::cout << "calc iir params\n";
		TIIRCoeff iirCoeffs = CalcIIRFilterCoeff(iirParams);
		// will use hfunction for sum all those polynoms to one combined one   
		//std::cout << "generate hfunction\n";
		// sections filter assign and allocate
		HFunction *sectionFilters = reshapeIIRFilter(iirCoeffs); // build section filter to multiplicate
		_ac_time_filter.multiplicateHFunctions(sectionFilters, iirCoeffs.NumSections); // multiplicate the  sections
		delete[] sectionFilters;
		//std::cout << "generate hfunction\n";
		if (params->show_filter) {
			std::cout << "Butterworth Analyze: Fc= " << (OmegaC*NyQuistFrequency) << ",Epsilon = " << epsilon << ",FilterOrder =" << filterOrder << "Num Sections = " << iirCoeffs.NumSections << "\n";
			std::cout << "Filter_Type = " << params->filterName << "\nOrder_Of_Filter = " << filterOrder << "\nFilter Structure=Direct Form I\nUnits=normalized(0 to 1)\n";
			_ac_time_filter.view();
		}
	}

	else {
		std::cout << "filter " << params->filterName << " is unsupported abort";
		throw std::runtime_error("Unable to create unsupported filter - 1");
	}
}