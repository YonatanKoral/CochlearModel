#pragma once
#include <vector>
#include <type_traits>
#include <algorithm>		
#include <functional>
#include <cmath>
#include <sstream>
# include <iostream>
#include <locale>         // std::locale, std::tolower
#include <regex>
#include <map>
#include "mutual.h"	
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#ifdef CUDA_MEX_PROJECT
#include <mex.h> 
#endif

#ifdef VS2013
#define PrintFormat printf
#endif
#ifndef VS2013
template <typename ... Args>
void PrintFormat(char const * const format,
	Args const & ... args) noexcept
{
	printf(format, args ...);
}
#endif
std::string transformString(const std::string& input, std::function<char(char, const std::locale)> tr);
std::string getFileType(const std::string& fileName);
std::string converToLower(const std::string& input);
std::string converToUpper(const std::string& input);
std::vector<std::string> splitToVector(const std::string& str, const std::string& regextest);
std::vector<const char*> createCharMatrix(const std::vector<std::string>& v);
template<typename T, typename V> struct mapper {
	std::map<T, V> _data;
	std::map<std::string, std::string> _names;
	mapper() {
		_data = std::map<T, V>();
		_names = std::map<std::string, std::string>();
	}
	void add(const T& t, const V& v) {
		_data.insert(std::map<T, V>::value_type(t, v));
	}
	void add(const std::string& t, const std::string& v) {
		_names.insert(std::map<std::string, std::string>::value_type(t, v));
	}
	std::map<T, V> getData() { return _data; }
	const std::map<T, V> getData() const { return _data; }
	std::map<std::string, std::string> getNames() { return _names; }
	const std::map<std::string, std::string> getNames() const { return _names; }
};



/*
template<typename T>
std::string outputString(std::ostringstream& oss, T t) {
	// recursive variadic function
	oss << t;
	return oss.str();
}
template<typename T, typename... Args>
std::string outputString(std::ostringstream& oss, T t, Args... args) // recursive variadic function
{
	
	return outputString(oss, args ...);
}

template<typename T, typename... Args>
std::string printString(T t, Args... args) // recursive variadic function
{
	std::ostringstream oss("");
	return outputString(oss, t, args);
}					 */