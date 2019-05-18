#include "HFunction.h"

void HFunction::load(const HFunction& h) {
	HFunction::load(h.Numerator, h.Denominator);
}
void HFunction::load(const vector<double>& Numerator, const vector<double>& Denominator) {
	//cout << "loading Numerator of size " << Numerator.size() << "\n";
	this->Numerator = Numerator;
	//cout << "loading Denominator of size " << Denominator.size() << "\n";
	this->Denominator = Denominator;
	//cout << "loaded hfunction \n";
}
HFunction::HFunction(const vector<double>& Numerator, const vector<double>& Denominator) :
Numerator(Numerator),
Denominator(Denominator)
{
	//cout << "loading hfunction nsize=" << Numerator.size() <<",dsize="<< Denominator.size()<<"\n";
	//load(Numerator,Denominator);
	//cout << "loaded hfunction nsize=" << Numerator.size() << ",dsize=" << Denominator.size() << "\n";
}
HFunction::HFunction() :
Numerator(1, 1),
Denominator(1, 1)
{
}
HFunction::HFunction(const int size) :
		Numerator(size,1),
		Denominator(size,1)
{
	//cout << "constructed hfunction nsize=" << Numerator.size() << ",dsize=" << Denominator.size() << "\n";
}

HFunction::HFunction(const HFunction& h) :
Numerator(h.Numerator),
Denominator(h.Denominator) {
	//cout << "copying hfunction \n";
	//HFunction::load(h);
	//cout << "constructed hfunction \n";
}
bool HFunction::isFIR() const {
	return Denominator.size() == 1 && Denominator[0] == 1;
}
void HFunction::setFIRGain(const double& gain) {
	double currentGain = sum(Numerator);
	Numerator = (pow(10,gain/20) / currentGain)*Numerator;
}
void HFunction::view() {
	view(0);
}

void HFunction::view(int Fs) {
	cout << "Numerator = [" << viewVector(Numerator) << "]\n";
	cout << "Denominator = [" << viewVector(Denominator) << "]\n";
	cout << "fvtool([" << viewVector(Numerator) << "]," << (Numerator.size() > 10 ? "... \n" : "") << "[" << viewVector(Denominator) << "]";
	if (Fs > 0) {
		cout << ",'Fs',"<< Fs;
	}
	cout << "); \n";
}
HFunction operator +(const HFunction& h_left, const HFunction& h_right) {
	vector<double> NumeratorLeft(conv(h_left.Numerator, h_right.Denominator));
	vector<double> NumeratorRight(conv(h_right.Numerator, h_left.Denominator));
	if (NumeratorLeft.size() < NumeratorRight.size()) NumeratorLeft.resize(NumeratorRight.size());
	else if (NumeratorRight.size() < NumeratorLeft.size()) NumeratorRight.resize(NumeratorLeft.size());
	// since convolve ensure us given size of vector per given size of inputs I can convolve to known given size (max_numerator_size+max_denominator_size-1) on both factor
	vector<double> Numerator(NumeratorLeft + NumeratorRight);
	truncZeros(Numerator);
	vector<double> Denominator(conv(h_left.Denominator, h_right.Denominator));
	truncZeros(Denominator);
	return HFunction(Numerator, Denominator);
}

HFunction& HFunction::operator *= (const HFunction& h) {
	Numerator = conv(Numerator, h.Numerator);
	truncZeros(Numerator);
	Denominator = conv(Denominator, h.Denominator);
	truncZeros(Denominator);
	return *this;
}

//HFunction operator *(const HFunction& h_left, const HFunction& h_right) {
//	// multiplying function meaning convolve numerator and denonatior
//
//	CVector Numerator = conv(h_left.Numerator, h_right.Numerator);
//	Numerator.truncZeros();
//	CVector Denominator = conv(h_left.Denominator, h_right.Denominator);
//	Denominator.truncZeros();
//	return HFunction(Numerator, Denominator);
//}

HFunction *reshapeIIRFilter(const TIIRCoeff& iirCoeffs) {
	HFunction *filtersArray = new HFunction[iirCoeffs.NumSections];
	for (int i = 0; i < iirCoeffs.NumSections; i++) {
		//cout << "handling section " << i << "\n";
		filtersArray[i].reshapeIIRFilterSection(iirCoeffs, i);
		//cout << "handled section " << i << "\n";
	}
	return filtersArray;
}
void HFunction::reshapeIIRFilterSection(const TIIRCoeff& iirCoeffs, const int section_number){
	Numerator.assign(5, 0);
	Denominator.assign(5, 0);
	//cout << "reshape section hfunction created\n";
	Numerator[0] = iirCoeffs.b0[section_number];
	Numerator[1] = iirCoeffs.b1[section_number];
	Numerator[2] = iirCoeffs.b2[section_number];
	Numerator[3] = iirCoeffs.b3[section_number];
	Numerator[4] = iirCoeffs.b4[section_number];
	Denominator[0] = iirCoeffs.a0[section_number];
	Denominator[1] = iirCoeffs.a1[section_number];
	Denominator[2] = iirCoeffs.a2[section_number];
	Denominator[3] = iirCoeffs.a3[section_number];
	Denominator[4] = iirCoeffs.a4[section_number];
	//cout << "IIR Section #" << section_number << "\n";
	//tfunction.view();
}
void HFunction::multiplicateHFunctions(const HFunction  *hfunctionsArray, const int section_number) {
	load(hfunctionsArray[0]); // summary starts from index 0 funnction
	//cout << "Starts summary section\n";
	//summary.view();
	for (int i = 1; i < section_number; i++) {
		(*this) *= hfunctionsArray[i];
		//cout << "Summary of "<< (i+1) <<" sections\n";
		//summary.view();
	}
}
/*
HFunction& HFunction::operator = (const HFunction& h) {
	Numerator.copy(h.Numerator);
	Denominator.copy(h.Denominator);
	return *this;
}
*/



void HFunction::decodeBinFile(const vector<double>& binBuffer) {
	int filterOrder = static_cast<int>(binBuffer[0]);
	bool isFIR = static_cast<int>(binBuffer[1]) == 1;
	Numerator.resize(filterOrder);
	if (isFIR) {
		Denominator.resize(1);
		Denominator.at(0) = 1;
	}
	else {
		Denominator.resize(filterOrder);
	}


	for (int i = 0; i < filterOrder; i++) {
		if (!isFIR) Denominator[i] = binBuffer[i + 2];
		Numerator[i] = binBuffer[i + 2+ (isFIR?0:filterOrder)];
	}
}
HFunction createFIRFunction(const double *input,int size){
	HFunction hfun(vector<double>(size, 0), vector<double>(1, 1));
	for (int i = 0; i < size; i++) {
		hfun.Numerator[i] = input[i];
	}
	return hfun;
}
HFunction createFIRFunction(PMOutput& pmo) {
	int size = static_cast<int>(pmo.h.size());
	HFunction hfun(vector<double>(size, 0), vector<double>(1, 1));
	for (int i = 0; i < size; i++) {
		hfun.Numerator[i] = pmo.h[i];
	}
	return hfun;
}
HFunction::~HFunction()
{
}
