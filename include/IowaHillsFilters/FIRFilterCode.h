//---------------------------------------------------------------------------

#ifndef FIRFilterCodeH
#define FIRFilterCodeH
#include "FFTCode.h"   // For the definition of TWindowType
#include <cmath>
//---------------------------------------------------------------------------
#include "../const.h"
#include "TFIRPassTypes.h"
 

 void FilterWithFIR(double *FirCoeff, int NumTaps, double *Signal, double *FilteredSignal, int NumSigPts);
 void FilterWithFIR2(double *FirCoeff, int NumTaps, double *Signal, double *FilteredSignal, int NumSigPts);
 void RectWinFIR(double *FirCoeff, int NumTaps, TFIRPassTypes PassType, double OmegaC, double BW);
 void WindowData(double *Data, int N, TWindowType WindowType, double Alpha, double Beta, bool UnityGain);
 double Sinc(double x);
 void FIRFreqError(double *Coeff, int NumTaps, int PassType, double *OmegaC, double *BW,double targetCornerGain=3);
 void FIRFilterWindow(double *FIRCoeff, int N, TWindowType WindowType, double Beta);
 void AdjustDelay(double *FirCoeff, int NumTaps, double Delay);
#endif
