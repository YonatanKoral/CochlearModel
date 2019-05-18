#ifndef HFUNCTION_H
#define HFUNCTION_H
#pragma once
#include "cvector.h"  
#include "IowaHillsFilters\IIRFilterCode.h"
#include "pm.h"
/**
* represents a H transfer function of division of to polynom each polynom represent by vector of its coefficents from low power to high such that
* poly = (A+B*x+C*x^2)/(D+E*x+F*x^2) will be represented by
* Numerator = [A B C], Denominator = [D E F]
*/
class HFunction
{
public:

	void load(const vector<double>& Numerator, const vector<double>& Denominator);
	void load(const HFunction& h);
	HFunction(const vector<double>& Numerator, const vector<double>& Denominator);
	HFunction();
	HFunction(const int size);
	~HFunction();
	void view(); // show represntation of the h function
	void view(int Fs); // show represntation of the h function
	vector<double> Numerator;
	vector<double> Denominator;
	HFunction(const HFunction& h);
	HFunction& operator *=(const HFunction& h);
	//HFunction& operator=(const HFunction& h);
	bool isFIR() const;

	/**
	*	set FIR gain, input is DB
	*/
	void setFIRGain(const double& gain);

	/**
	* assigning section_number coeffs to filter
	*/
	void reshapeIIRFilterSection(const TIIRCoeff& iirCoeffs, const int section_number);

	/**
	* assigning multiplication array to the hfunction
	*/
	void multiplicateHFunctions(const HFunction *hfunctionsArray, const int section_number);

	/**
	* decodes bin file filter array to h function with appropriat numratot and denominator
	*/
	void decodeBinFile(const vector<double>& binBuffer);
};
HFunction operator +(const HFunction& v_left, const HFunction& v_right);
//HFunction operator *(const HFunction& v_left, const HFunction& v_right);
HFunction *reshapeIIRFilter(const TIIRCoeff& iirCoeffs);

HFunction createFIRFunction(const double *input,int size);
HFunction createFIRFunction(PMOutput& pmo);
#endif
