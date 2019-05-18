
# pragma once

template <typename T> class vector {	

	T *coeff;
	int len;

public:
	
	vector(int length);
	vector (int length, T init_val);
	vector vector::operator=(const vector & rhs);
};