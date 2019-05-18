#pragma once
#ifndef __COMPLEXJNDPROFILE__H
#define __COMPLEXJNDPROFILE__H
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
# include <cmath>
# include "const.h"
# include <map>
# include "cvector.h"
# include "cochlea_common.h"
class ComplexJNDProfile {
public:

	float _frequency; //in HZ
	float _dBSPLNoise; 	 // the noise level in dBSPL
	vector<int> _intervals;
	float  _minValueFound;
	std::string _signal_name;
	void setInterval(int position, int index);
	std::vector<double>	getRelevantValues(const std::vector<double>& all_values, const std::vector<int>& indexes);
	float calculateMinValue(const vector<double>& rawJNDValues, const vector<int>& Failed_Signal_Indexes, float eps, bool view_parts);
	float calculateGradientMinMaxValue(const vector<double>& rawJNDValues, const vector<int>& Failed_Signal_Indexes, float eps, bool view_parts);
	int calculateMinValueWarning;
	int calculateGradientMinMaxValueWarning;
	int Failed_Convergence_Warning;
	std::string viewCaptured(const double& minValueFound,const vector<double>& tested_values,const vector<double>& minimums_captured);
	void updateBaseValues(float frequency, float dBSPLNoise, int length, const std::string &signal_name);
	std::string show(int index);
	std::string showLegend(const int&);
	ComplexJNDProfile(float frequency, float dBSPLNoise, int length,const std::string &signal_name);
	ComplexJNDProfile();
	~ComplexJNDProfile();
};
#endif

