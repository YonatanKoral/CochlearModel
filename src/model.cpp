/******************************************************************\

	File Name :		model.cpp	
	===========

	Classes Implemented :	CModel
	=====================

	Description	: a complete model of the cochlea
	=============


\******************************************************************/

#include "model.h"

// c'tor - init all values with defaults
CModel::CModel(CParams *tparams,int params_counter) : 
	_sections( SECTIONS ),			// [] number of partitition sections of the cochlear length.
	_dx( COCHLEA_LEN/SECTIONS),		// [cm] cochlear length partitition distance.		
	_area( COCHLEA_AREA ),			// [cm^2] cochlear cross section
	_len( COCHLEA_LEN ),			// [cm] cochlear length
	_rho( LIQUID_RHO ),				// [g/cm^3] liquid density
	_beta( BM_WIDTH ),				// [cm] non-linear width constants
	_w_ohc( W_OHC),					// [rad] OHC resonance frequency.
	_psi0( OHC_PSI_0 ),				// [V] resting OHC potential.		
    _eta_1( ETA_1 ),				// [V*sec/m] Electrical features of the hair cell.
    _eta_2( ETA_2 ),				// [V/m] Electrical features of the hair cell.
	_ratio_K( RATIO_K ),			// [] ratio between the TM stiffness and the OHC stifness	
	_alpha_s( ALPHA_S ),			// _alpha_s & _alpha_l are two arbitrary constants used in the OHC's nonlinearity (for the tanh(x) function).
	_alpha_L( ALPHA_L ),			// _alpha_s & _alpha_l are two arbitrary constants used in the OHC's nonlinearity (for the tanh(x) function).
	_alpha_r( BM_ALPHA_R ),			// _alpha_r - Linear factor, in the BM equation: P_BM = ... + (1 + alpha_r*csi_BM^2)*csi_BM + ...
	_w_ow( W_OW ),					// [rad] Middle ear frequency.
	_w_ow_pow2( W_OW*W_OW ),		// [rad^2] _w_ow^2.
	_sigma_ow( SIGMA_OW ),			// [gr/cm^2]Oval window aerial density	
	_gamma_ow( GAMMA_OW ),			// [1/sec] Middle ear damping constant
	_Cme( C_ME ),					// Mechanical gain of the ossicles
	_Gme( G_ME ),					// Coupling of oval window displacement to ear canal pressure
	_Cow( C_OW ),					// Coupling of oval window to basilar membrane
	_OW_flag( OW_FLAG ),
	_OHC_NL_flag( OHC_NL_FLAG ),				// Enable\Disable the OHC nonlinearity				
	_active_gamma_flag( ACTIVE_GAMMA_FLAG ),	// Enable\Disable the change of gamma at float time, according to <psi>.
	_a0( BC_0 ),								// Boundary conditions - matrix
	_a1( BC_1 ),
	_a2( BC_2 ),
	_params_counter(params_counter),
	// BM
	_M_bm(_sections, 0),		// mass vector
	_R_bm(_sections, 0),		// restrain vector
	_S_bm(_sections, 0),		// elasticity vector

	// TM
	_M_tm(_sections, 0),
	_R_tm(_sections, 0),		// restrain vector
	_S_tm(_sections, 0),		// elasticity vector

	// OHC
	_S_ohc(_sections, 0),		// elasticity vector

	// OC
	_w_cf_pow2(_sections, 0),	// The (approximately) OC resonance vector
	_Q(_sections,0)
{
	//cout << "loading configuration..." << endl;
	configurations.reserve(this->_params_counter);
	for ( int i = 0;i<this->_params_counter;i++) {
		configurations.push_back(SubModel(&tparams[i], _sections));
	}	
	//cout << "finished loading configuration..." << endl;

	// Assign values to the model's vectors:
	for (int i=0; i< _sections; ++i)
	{
		_M_bm[i] =	M0 * exp( M1 * _dx * i );
		_R_bm[i] =	R0 * exp( R1 * _dx * i );
		_S_bm[i] =	S0 * exp(S1Cochlear * _dx * i );

		_w_cf_pow2[i] = _S_bm[i]/_M_bm[i];

		_R_tm[i] =	_R_bm[i];
		_S_tm[i] =	_R_bm[i]*_w_cf_pow2[i]/_w_ohc;

		_S_ohc[i] =	_ratio_K*_S_tm[i];
		// A constant vector: 2*rho*beta/(area*_M_bm)
		_Q[i] = 2 * _rho*_beta / (_area * _M_bm[i]);
	}	
	
	

}

// finds first param set with enabled lambda
int CModel::firstLambdaEnabled() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->disable_lambda==false) {
			res = i;
			break;
		}
	}
	//cout << "first lambda enabled is: " << res << "\n";
	return res;
}

// finds first param set with CPU Lambda calculation enforced
int CModel::firstForceCPUOnIHC() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->run_ihc_on_cpu == true && configurations[i].params->disable_lambda == false) {
			res = i;
			break;
		}
	}
	//cout << "first force CPU enabled is: " << res << "\n";
	return res;
}


// finds first param set with CPU Lambda calculation enforced
int CModel::firstLambdaCUDARun() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->run_ihc_on_cpu == false && configurations[i].params->disable_lambda == false) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}

int CModel::firstJNDCalculationSet() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if ( configurations[i].params->Calculate_JND == true) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}

int CModel::firstCartesicMultiplicationOfInputsSet() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->Generating_Input_Profile()) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}

int CModel::firstJNDCalculationSetONGPU() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->Calculate_JND == true && configurations[i].params->Calculate_JND_On_CPU == false) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}


int CModel::firstJNDCalculationSetONCPU() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->Calculate_JND == true && configurations[i].params->Calculate_JND_On_CPU == true) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}

int CModel::firstGeneratedInputSet() {
	int res = PARAM_SET_NOT_EXIST;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->generatedInputProfile ) {
			res = i;
			break;
		}
	}
	//cout << "first GPU Lambda enabled is: " << res << "\n";
	return res;
}

int CModel::maxJNDIntervals() {
	int size = 0;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->Generating_Input_Profile() && configurations[i].params->numberOfJNDIntervals() > size ) {
			size = configurations[i].params->numberOfJNDIntervals();
		}
	}
	return size;
}
/*
int CModel::maxBackupNodesLength() {
	int res = 0;
	for (int i = 0; i < static_cast<int>(configurations.size()); i++) {
		if (configurations[i].params->totalNodesLastBlockSaved()>res) {
			res = configurations[i].params->totalNodesLastBlockSaved();
		}
	}
	//cout << "max backup nodes length " << res << "\n";
	return res;
}
*/

void CModel::freeModel() {

	//cout << "Free The Model!!!\n";
	
}
// d'tor
CModel::~CModel()	{
	//cout << "Destroying model!!!\n";
	freeModel();
}
