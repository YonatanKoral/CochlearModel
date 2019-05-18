/******************************************************************\

	File Name :		tridiag.h
	===========

	Classes Defined :	CTriDiag
	=================

	Description	:	Solve a tri-diagonal matrix 
	=============


\******************************************************************/
#pragma once

#ifndef __CTRIDIAG
#define __CTRIDIAG

#include "mutual.h"
#include "cvector.h"

// holds the 3 diagonal matrix, and solve for different vectors

class CTriDiag
{
	vector<double>	_k, _h;		// temporary value vectors

	// flags = initialized diagonal
	bool _l_init, _m_init, _u_init;

//private :

	void ReCalc(void);		// prepare system for solving
	bool Init(void) const;	

public:

	vector<double>	_l, _m, _u; // lower, mid, and upper diagonals
	CTriDiag(int sections); // c'tor
	
	// Inits with lower, mid, and upper diagonal, respectively
	// on lower and upper, last element is meaningless
	void SetMidDiag(const vector<double>& diag_m);
	void SetLowerDiag(const vector<double>& diag_l);
	void SetUpperDiag(const vector<double>& diag_u);

	// solving for this vector
	vector<double> SolveFor(const vector<double>& v);

};

#endif