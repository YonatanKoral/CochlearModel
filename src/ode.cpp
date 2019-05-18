

#include "ode.h"

ode::ode() {}

// Implement euler method predictor ( 1 time step )
vector<double> ode::Euler(const vector<double>& V_past, const vector<double>& dV_past, const double& dt_now)
{
	return ( dt_now * dV_past + V_past );				// ==> V(t+dt)
}

double ode::Euler( const double& s_past, const double& ds_past, const double& dt_now )
{
	return ( dt_now * ds_past + s_past );				// ==> V(t+dt)
}

// Implement euler method predictor ( 1 time step )
vector<double> ode::Trapezoidal(const vector<double>& V_past, const vector<double>& dV_past, const vector<double>& dV_now, const double& dt_now)
{

	return ( V_past + 0.5 * dt_now * (dV_past + dV_now) );		// ==> V(t+dt)
}

double ode::Trapezoidal( const double& s_past, const double& ds_past, const double& ds_now, const double& dt_now )
{
	return ( s_past + 0.5 * dt_now * (ds_past + ds_now) );		// ==> V(t+dt)

}


// used the current state now, the current step size
// and past values of membrane speed (past_sp) and acceleration
// (past_acc). The past values are taken before the last correction 
bool ode::Tester(
	const vector<double>& past_sp,
	const vector<double>& past_acc,
	const vector<double>& now_sp,
	const vector<double>& now_acc,
	double& step, 
	bool& another_loop,
	bool& is_next_time_step
	)
{

	////double max_m1_sp_err;
	//bool another_loop;

	// Lipschitz condition number
	double	Lipschitz;			

	// find the speed error (compared to the past value)
	vector<double> sp_err = abs(past_sp - now_sp);
	
	// prepare a speed error limit vector
	// the value is the local maximum between the constant tolerance
	// and the local spped * step size
	vector<double> sp_err_limit = vmax<double>(static_cast<double>(MAX_TOLERANCE), abs(now_sp*step));
	
	// 1st moment of speed error vector
	double m1_sp_err = sum<double>( sp_err );  //udish fix bug abs 25/11/03
	
	// counts the number of points in which the error is greater than
	// the limit
	double fault_counter = count_greater( sp_err, sp_err_limit );
	
	// calculate lipschitz number 
	if ( m1_sp_err > MAX_M1_SP_ERROR ) //(m1_sp_err > max_m1_sp_err)
		Lipschitz = vmax<double>( abs( past_acc - now_acc ) ) / m1_sp_err;
	else
		Lipschitz = 0;


	// use the calculated values to decide
	if ( Lipschitz * step > 2.0 )
	{
		// iteration didn't pass, step size decreased
		step *= 0.5;
		another_loop = false;				// another == false
		is_next_time_step = false;			
	}
	else 
		if ( fault_counter > 0 )
		{
			// another iteration is needed for this step
			step *= 1.0;			
			another_loop = true;			// another == true
			is_next_time_step = false;
		}
		else 
			if (Lipschitz * step < 0.25 )
			{
				// iteration passed, step size increased
				step *= 2.0;
				another_loop = false;		// another == false. Move on to next time step.
				is_next_time_step = true;
			}
			else 
			{
				// iteration passed, same step size
				step *= 1.0;
				another_loop = false;		// another == false. Move on to next time step.
				is_next_time_step = true;
			}

	return another_loop;

}

// count the number of positions in which elements in v1
// are greater than their corresponding v2 elements
int ode::count_greater(const vector<double>& v1, const vector<double>& v2)
{
	int cnt = 0;
	for ( int i=0; i < (int)v2.size(); i++)
		cnt += (v1[i] > v2[i]);
	return cnt;
}