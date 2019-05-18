/******************************************************************\

	File Name :		CV.cpp
	===========

	Classes Implemented :	CTriDiag
	=====================

	Description	:	various vector functions required
	=============

	Notes : 
	=======
		1. I'm using the <std::vector> inheritance in the operators overloading.
		2. Avoid using std::vector with integers (scalars) when expecting a double, e.g:
		   std::vector * 1/2 OR 1/2 * std::vector. The multiplication in this case will fail !
		   Try std::vector * 1/2.0 OR std::vector * 0.5 insteed.


\******************************************************************/
#pragma once

#ifndef __CVECTOR
#define __CVECTOR
#include <vector>
#include <map>
#include <type_traits>
#include <algorithm>		
#include <functional>
#include <cmath>
#include <sstream>
# include <iostream>
#include <locale>         // std::locale, std::tolower
#include <regex>
#include "mutual.h"
typedef struct viewVectorModes {
	int line_seperation_every_x_values; // line seperations when display std::vector
	std::ios_base::fmtflags float_flags;
	int precision; // float precision
} viewVectorModes;
// retrieve std::vector of given indices
template<class T> struct indexLocatorPredicate {
	const std::vector<int> _indexes;
	indexLocatorPredicate(const std::vector<int>& indexes) : _indexes(indexes) {}
	std::vector<T, std::allocator<T>> operator() (const std::vector<T, std::allocator<T>>& v) {
		std::vector<T, std::allocator<T>> r;
		for (auto i : _indexes) {
			r.push_back(v[i]);
		}
		return r;
	}
};
//class CVector : private std::std::vector<double>
//{
//
//public:
//	static long allocated_size;
//	static long released_size;
//	static long allocated_times;
//	static long released_times;
//	using std::vector::push_back;
//	using std::vector::operator[];
//	using std::vector::begin;
//	using std::vector::end;
//	using std::vector::assign;
//	// c'tor
//	CVector(const CVector& v); // copy constructor
//	CVector(const int std::vector_length);
//	CVector(const int std::vector_length, const double value);
//	std::string view();
//	CVector& operator +=(const double scalar);
//	CVector& operator -=(const double scalar);
//	//CVector tanh(const CVector v);	
//
//	// destructor
//	virtual ~CVector(void);
//
//	/**
//	*	if min_size is larger than the std::vector size	it will add trailing zeros
//	*	otherwise it will not changed
//	*/
//	void expand(const int& min_size);
//
//
//
//	/**
//	* trunc all trailing zeros at the end of std::vector by removing all data at the end of the std::vector that is zeros
//	*/
//	void truncZeros();
//
//	/**
//	* cloning the std::vector v to this std::vector by resizing and than copy all elements
//	*/
//	void copy(const CVector &v);
//
//};
//
//CVector operator +(const CVector& v_left, const CVector& v_right);
//CVector operator +(const CVector& v_left, const double& scalar);
//CVector operator +(const double& scalar, const CVector& v_right);
//
//CVector operator -(const CVector& v_left, const CVector& v_right);
//CVector operator -(const CVector& v_left, const double& scalar);
//CVector operator -(const double& scalar, const CVector& v_right);
//
//CVector operator *(const CVector& v_left, const CVector& v_right);
//CVector operator *(const CVector& v_left, const double& scalar);
//CVector operator *(const double& scalar, const CVector& v_right);
//
//CVector operator /(const CVector& v_left, const CVector& v_right);
//CVector operator /(const CVector& v_left, const double& scalar);
//CVector operator /(const double& scalar, const CVector& v_right);
//
//// Overloading the exp() function for the CVector class
//template<class T,class std::allocator<T>> std::vector<T, std::allocator<T>> exp(const std::vector<T, std::allocator<T>>& v);
//
//double vmax(const CVector& v);									// NOTE: returns a <double> !
//CVector vmax(const CVector& v_left, const CVector& v_right);
//CVector vmax(const T& scalar, const CVector& v);
//CVector vmax(const CVector& v, const double& scalar);
// 
typedef struct simplifiedData {
	size_t length;
	void *data;
} simplifiedData;

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type{};
template<class T> auto operator +(const std::vector<T, std::allocator<T>>& v_left, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;
template<class T> auto operator +(const std::vector<T, std::allocator<T>>& v_left, const T& scalar)->std::vector<T, std::allocator<T>>;
template<class T> auto operator +(const T& scalar, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;

template<class T> auto operator -(const std::vector<T, std::allocator<T>>& v_left, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;
template<class T> auto operator -(const std::vector<T, std::allocator<T>>& v_left, const T& scalar)->std::vector<T, std::allocator<T>>;
template<class T> auto operator -(const T& scalar, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;

template<class T> auto operator *(const std::vector<T, std::allocator<T>>& v_left, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;
template<class T> auto operator *(const std::vector<T, std::allocator<T>>& v_left, const T& scalar)->std::vector<T, std::allocator<T>>;
template<class T> auto operator *(const T& scalar, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;

template<class T> auto operator /(const std::vector<T, std::allocator<T>>& v_left, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;
template<class T> auto operator /(const std::vector<T, std::allocator<T>>& v_left, const T& scalar)->std::vector<T, std::allocator<T>>;
template<class T> auto operator / (const T& scalar, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;

// Overloading the exp() function for the CVector class
template<class T> auto exp(const std::vector<T, std::allocator<T>>& v)->std::vector<T, std::allocator<T>>;

template<class T> auto vmax(const std::vector<T, std::allocator<T>>& v) -> T;									// NOTE: returns a <double> !
template<class T> auto vmax(const std::vector<T, std::allocator<T>>& v_left, const std::vector<T, std::allocator<T>>& v_right)->std::vector<T, std::allocator<T>>;
template<class T> auto vmax(const T& scalar, const std::vector<T, std::allocator<T>>& v)->std::vector<T, std::allocator<T>>;
template<class T> auto vmax(const std::vector<T, std::allocator<T>>& v, const T& scalar)->std::vector<T, std::allocator<T>>;
template<class T> void truncZeros(std::vector<T, std::allocator<T>>& v);
template<class T> auto get(const std::vector<T, std::allocator<T>>& v, const int& i) -> T;
template<class T> auto sum(const std::vector<T, std::allocator<T>>& v) -> T;
template<class T,class T2> auto partialSum(const std::vector<T, std::allocator<T>>& v,const int& target_size)->std::vector<T2, std::allocator<T2>>;
template<class T, class T2> auto partialAvg(const std::vector<T, std::allocator<T>>& v, const int& target_size)->std::vector<T2, std::allocator<T2>>;
template<class T> auto conv(const std::vector<T, std::allocator<T>>& A, const std::vector<T, std::allocator<T>>& B)->std::vector<T, std::allocator<T>>;
template<class T> auto abs(const std::vector<T, std::allocator<T>>& v)->std::vector<T, std::allocator<T>>;
//template<class T> void self_add(std::vector<T, std::allocator<T>>& v, const size_t& start_index, const size_t& end_index, const T& value);
template<class T> auto operator ^(const std::vector<T, std::allocator<T>>& v, const T& p)->std::vector<T, std::allocator<T>>;
template<class T> auto sqrt(const std::vector<T, std::allocator<T>>& v)->std::vector<T, std::allocator<T>>;
template<class T> auto tanh(const std::vector<T, std::allocator<T>>& v)->std::vector<T, std::allocator<T>>;
//template<class T> auto stepVector(const std::vector<T, std::allocator<T>>& v, const int& step, const int& first_index)->std::vector<T, std::allocator<T>>;
template<class T> std::string viewVector(const std::vector<T, std::allocator<T>>& v, const viewVectorModes& modes);
template<class T> std::string viewVector(const std::vector<T, std::allocator<T>>& v, const int& line_seperation_every_x_values);
template<class T> std::string viewVector(const std::vector<T, std::allocator<T>>& v);
template<class T> bool checkVector(const std::vector<T, std::allocator<T>>& v,const T& value);
template <class T> auto parseToScalar(const std::string &Text)->T;
template <typename T> auto lexicalCast(T Number)->std::string;
template<class T> auto parseToVector(const std::string& str)->std::vector<T, std::allocator<T>>;
template<class T1, class T2> struct CastToT2 {
	T2 operator()(T1 value) const { return static_cast<T2>(value); }
};
template<class T> auto replicateVector(const std::vector<T, std::allocator<T>>& v,const size_t& times)->std::vector<T, std::allocator<T>>;
template<class T> auto expandVector(const std::vector<T, std::allocator<T>>& v, const size_t& times)->std::vector<T, std::allocator<T>>;
template<class T> auto expandVectorToSize(const std::vector<T, std::allocator<T>>& v, const size_t& to_size)->std::vector<T, std::allocator<T>>;
template<class T> auto replicateAndxpandVector(const std::vector<T, std::allocator<T>>& v,const size_t& times_replicate, const size_t& times_expand)->std::vector<T, std::allocator<T>>;
template<class T, class T2> std::string verboseVectors(const std::string& prefix, const int& precision, const int& seperation_frequency, const std::vector<T, std::allocator<T>>& v, const std::vector<T2, std::allocator<T2>>& v2);
template<class T1, class T2, class T3> void vectorSumTemplate(const std::vector<T1>& A, const T1& c1, const std::vector<T2>& B, const T2& c2, std::vector<T3>& C);
template<class T1, class T2> auto castVector(const std::vector<T1, std::allocator<T1>>& v)->std::vector<T2, std::allocator<T2>>;
template<class T1, class T2> auto castVector(const T1 *v,size_t vsize)->std::vector<T2, std::allocator<T2>>;
template<class T1, class T2> auto castVector(const simplifiedData& dtv)->std::vector<T2, std::allocator<T2>> { return castVector<T1, T2>((T1 *)dtv.data,dtv.length); }
template<class T1, class T2, class T3> void FIRFilterTemplate(const std::vector<T1>& X, std::vector<T2>& Y, const std::vector<T3>& filter, int sections, int lambdaOffset, size_t time_dimension, int starts);
template<class T1, class T2, class T3> void IIRFilterTemplate(const std::vector<T1>& X, std::vector<T2>& Y, const std::vector<T3>& a, const std::vector<T3>& b, int sections, int lambdaOffset, size_t time_dimension, int starts);
template<class T> auto transposeVector(const std::vector<T, std::allocator<T>>& v, const size_t& column_size)->std::vector<T, std::allocator<T>>;
template<class T> auto transposeVector(const std::vector<T, std::allocator<T>>& v, const size_t& column_size, const size_t& partitions)->std::vector<T, std::allocator<T>>;
template< class keyClass, class valueClass > std::vector< keyClass > getAllKeys(const std::map< keyClass , valueClass > &m);
template< class keyClass, class valueClass > std::vector< valueClass > getAllValues(const std::map< keyClass, valueClass > &m);
inline void addToStream(std::ostringstream&) {}
template<typename T, typename... Args> void addToStream(std::ostringstream& a_stream, T&& a_value, Args&&... a_args);

template< typename... Args > std::string concat(Args&&... a_args);
template<typename T1,typename T2, typename PredicateT > void erase_if(std::map<T1,T2>& items, const PredicateT& predicate);

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	typedef std::integral_constant<size_t, 8> edge_size;
	typedef std::integral_constant<size_t, edge_size::value + 1> encapsulated_edge_size;
	typedef std::integral_constant<size_t, 2 * edge_size::value> double_edge_size;
	if (!v.empty()) {
		out << '[';
		if (v.size() < 20) {
			std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
		} else {
			std::copy(v.begin(), std::next(v.begin(), encapsulated_edge_size::value), std::ostream_iterator<T>(out, ", "));
			out << "... " << (v.size() - double_edge_size::value) << " more items ... ";
				std::copy(std::prev(v.end(), encapsulated_edge_size::value), v.end(), std::ostream_iterator<T>(out, ", "));
		}
		out << "\b\b]";
	}
	return out;
}


#include "cvector.cpp"
#endif




