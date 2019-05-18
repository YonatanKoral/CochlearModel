/******************************************************************\

	File Name :		tridiag.cpp
	===========

	Classes Implemented :	CTriDiag
	=====================

	Description	:	solves a tri-diagonal matrix
	=============


\******************************************************************/


#include "tridiag.h"

// c'tor
CTriDiag::CTriDiag(int sections) :
_l(sections, 0),
_m(sections, 0),
_u(sections, 0),
_k(sections, 0),
_h(sections, 0)
{
	_u_init = false;
	_m_init = false;
	_l_init = false;

	 //Init all vectors:
	

}


// set the upper diagonal. 
// last element in this vector is meaningless
void CTriDiag::SetUpperDiag(const vector<double>& diag_u)
{
	_u = diag_u;
	_u_init = true;
	if (Init())
		ReCalc();
}

// set the lower diagonal. 
// last element in this vector is meaningless
void CTriDiag::SetLowerDiag(const vector<double>& diag_l)
{
	_l = diag_l;
	_l_init = true;
	if (Init())
		ReCalc();
}

// set the middle diagonal. 
void CTriDiag::SetMidDiag(const vector<double>& diag_m)
{
	_m = diag_m;
	_m_init = true;
	if (Init())
		ReCalc();
}


	
void CTriDiag::ReCalc(void)
{
	// prepare vectors : 
	// see my document : tridag.doc for explanation

	_k[0] = _m[0];
	_h[0] = _u[0] / _k[0];
	
	for (int i=1; i< (int)_m.size() ; ++i)
	{
		_k[i] = _m[i] - _l[i-1] * _h[i-1];
		_h[i] = _u[i] / _k[i];
	}
}

// returns true only if all 3 diagonals are initialized
bool CTriDiag::Init(void) const
{
	return (_m_init && _l_init && _u_init);
}
// solves for this vector
vector<double> CTriDiag::SolveFor(const vector<double>& v)
{
	// see my document : tridag.doc for explanation
	//assert( v.size() == _k.size() );
	//assert(Init());
	vector<double> res(SECTIONS, 0), p(SECTIONS, 0);
	
	int n = static_cast<int>(v.size());
	
	p[0] = v[0] / _m[0];
	for (int i=1; i < n; ++i)
		p[i] = (v[i] - _l[i-1] * p[i-1]) / _k[i];
	
	res[n-1] = p[n-1];
	for (int j= n-2 ; j >= 0 ; --j)
		res[j] = p[j] - res[j+1] * _h[j];

	return res;
}

