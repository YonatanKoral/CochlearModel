/******************************************************************\

	File Name :		model.h
	===========

	Classes Defined :	CModel
	=================

	Description	: 
	=============
		A complete model of the cochlea. The clas holds all parameters that are 
		fixed to every step (not changing in time);

\******************************************************************/
#pragma once

#ifndef __MODEL
#define __MODEL

#include "mutual.h"
#include "params.h"
#include "cvector.h"
#include "bin.h"
#include "SubModel.h"
#include "IowaHillsFilters\NewParksMcClellan.h"
#include "IowaHillsFilters\IIRFilterCode.h"
#include "HFunction.h"
#include "firpm/pm.h"
#include "firpm/band.h"
#include "firpm/barycentric.h"
#include <set>
#include <fstream>
#include <sstream>
// the cochlear model
class CModel  
{

	//// prevent others from copying this class
	//CModel(const CModel& other); // hided copy c'tor
	//CModel& operator= (const CModel& other);// hided operator =

public:

	// Scalar model parameters:
	// ------------------------
	const int		_sections;			// [] number of partitition sections of the cochlear length.
	const double	_dx;				// [cm] cochlear length partitition distance.
	const double	_area;				// [cm^2] cochlear cross section.
	const double	_len;				// [cm] cochlear length.
	const double	_rho;				// [gr/cm^3] liquid density.
	const double	_beta;				// [cm] BM width.
	const double	_w_ohc;				// [rad] OHC resonance frequency.
	const double	_psi0;				// [V] resting OHC potential.	
    const double	_eta_1;				// [V*sec/m] Electrical features of the hair cell.
    const double	_eta_2;				// [V/m] Electrical features of the hair cell.
	
	const double	_ratio_K;			// [] ratio between the TM stiffness and the OHC stifness				
	const double	_alpha_s;			// Arbitrary constant (found by Dani Mackrants, 2007).
	const double	_alpha_L;			// Arbitrary constant (found by Dani Mackrants, 2007).
	const double    _alpha_r;			// _alpha_r - Linear factor, in the BM equation: P_BM = ... + (1 + alpha_r*csi_BM^2)*csi_BM + ...
	const double	_w_ow;				// [rad] Middle ear frequency.
	const double    _w_ow_pow2;			// [rad^2] w_ow^2.
	const double	_sigma_ow;			// [gr/cm^2]Oval window aerial density	
	const double	_gamma_ow;			// [1/sec] Middle ear damping constant
	const double	_Cme;				// Mechanical gain of the ossicles
	const double	_Gme;				// Coupling of oval window displacement to ear canal pressure
	const double	_Cow;				// Coupling of oval window to basilar membrane

	const double	_a0;				// Boundary conditions - matrix
	const double	_a1;				// Boundary conditions - vector Y
	const double	_a2;				// Boundary conditions - vector Y

	// Vector model parameters:
	// ------------------------
	// Flags
	const bool	_OW_flag;				// Enable\Disable the OW boundary conditions
	const bool	_OHC_NL_flag;			// Enable\Disable the OHC nonlinearity		
	const bool	_active_gamma_flag;		// Enable the change of gamma at float time, according to <psi>
	// BM  
	vector<double>		_M_bm;					// Basilar membrane mass density per unit area vector
	vector<double>		_R_bm;					// Basilar membrane resistance per unit area vector
	vector<double>		_S_bm;					// Basilar membrane stiffness per unit area vector

	// TM  
	vector<double>		_M_tm;					// Tectorial membrane mass density per unit area vector
	vector<double>		_R_tm;					// Tectorial membrane resistance per unit area vector
	vector<double>		_S_tm;					// Tectorial membrane stiffness per unit area vector

	// OHC  
	vector<double>		_S_ohc;					// OHC stiffness per unit area vector

	// OC (organ of corti)
	vector<double>		_Q;						// 2*rho*beta/(area*_M_bm)	
	vector<double>		_w_cf_pow2;				// The (approximately) OC resonance vector


	const int _params_counter;			// numbver of handeled params set
	vector<SubModel> configurations;
	int firstCartesicMultiplicationOfInputsSet(); // find first set with complex cartesic  input set
	int firstLambdaEnabled(); // first parameter set to not disable lambda, PARAM_SET_NOT_EXIST if none exist
	int firstForceCPUOnIHC(); // first parameter set to force ihc on cpu, PARAM_SET_NOT_EXIST if none exist
	int firstLambdaCUDARun(); // first parameter set to not force ihc on cpu, PARAM_SET_NOT_EXIST if none exist
	int firstJNDCalculationSet(); // first set thats calculates JND
	int firstJNDCalculationSetONGPU(); // first set thats calculates JND On GPU
	int firstJNDCalculationSetONCPU(); // first set thats calculates JND On GPU
	int maxJNDIntervals(); // max intervals on any model which JND calculated
	int firstGeneratedInputSet(); // returns first set of generated input if exists or PARAM_SET_NOT_EXIST otherwise
	//int maxBackupNodesLength(); // max length of backup nodes for audiogram definitions
	// Functions:
	// ----------
	CModel(CParams *tparams,int params_counter);
	void freeModel();
	~CModel();
	
};


#endif