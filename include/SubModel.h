#pragma once
#ifndef __SUBMODEL
#define __SUBMODEL

#include "mutual.h"
#include "params.h"
#include "cvector.h"
#include "error.h"
#include "bin.h"
#include "IowaHillsFilters\NewParksMcClellan.h"
#include "IowaHillsFilters\IIRFilterCode.h"
#include "HFunction.h"
#include "pm.h"
#include "band.h"
#include "barycentric.h"
#include "TBin.h"
#include <set>
#include <fstream>
#include <sstream>
class SubModel
{
public:
	CParams *params;	// sub model configurations params
	vector<double>		_gamma;					// [cm^-2] OHCs relative density (0.5=healthy)
	vector<double>		_nerves;					// [cm^-2] OHCs relative density (0.5=healthy)
	double _dbA; // sinus dbA power
	int _num_frequencies; // divided power number of frequncies
	const int _sections;
	HFunction		_ac_time_filter;
	HFunction		_noise_filter;
	SubModel(CParams *input_Params,int sections);
	SubModel(const SubModel &src); // copy constrcutor
	~SubModel();




	/**
	* calcs 10^(input/20) basically from DB to linear
	*/
	static double toLinear(double input);

	/**
	* calcs 10^(-1*input/20) basically from DB to linear but since delata come in reverse marks will negate it
	*/
	static double toDelta(double input);

	/**
	* calcs 20log10(input)converge to DB above 1
	*/
	static double toDB(double input);

	/**
	* using Oppenheim & Schafer 2nd addition digital signal processing book equation 7.104 at page 	502
	*/
	static int calcMinimumOrder(double delta1, double delta2, double transitionWidth);

	// use firpm library to calculate Park-Mecllelan filter from cannonic parameters
	PMOutput calcFIRPM(const double& OmegaC, const double& transitionWidth, const double& weightPass, const double& weightStop, const size_t& NumTaps);

	// use params member to calculate _ac_time_filter values
	void analyzeACFilter();
};

#endif

