/******************************************************************\

	File Name :		state.h
	===========

	Classes Defined :	CState, CSolutionMat
	=================

	Description	:	define a state vector, and the solution matrix
	=============
					which holds the calculated vectors

\******************************************************************/
#pragma once

#ifndef __ODE
#define __ODE

#include "mutual.h"
#include "cvector.h"

// Available File types (Log):
enum ODE_TYPE { EULER, TRAPEZIDAL };


class ode
{

public:

	//c'tor
	ode();

	// Implement euler method predictor ( 1 time step ) for all relevant parameters
	double Euler( const double& V_past, const double& dV_past, const double& dt_now );
	vector<double> Euler(const vector<double>& V_past, const vector<double>& dV_past, const double& dt_now);

	// Implement modified euler method predictor ( 1 time step ) for all relevant parameters
	double Trapezoidal( const double& V_past, const double& dV_past, const double& dV_now, const double& dt_now );
	vector<double> Trapezoidal(const vector<double>& V_past, const vector<double>& dV_past, const vector<double>& dV_now, const double& dt_now);

	// used the current state now, the current step size
	// and past values of membrane speed (past_sp) and acceleration
	// (past_acc). The past values are taken before the last correction 
	bool Tester(
		const vector<double>& past_sp,
		const vector<double>& past_acc,
		const vector<double>& now_sp,
		const vector<double>& now_acc,
		double& step, 
		bool& another_loop,
		bool& is_next_time_step
	);

	int count_greater(const vector<double>& v1, const vector<double>& v2);

};


#endif


