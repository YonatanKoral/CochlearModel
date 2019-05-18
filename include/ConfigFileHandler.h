#pragma once
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <sstream>
# include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <locale>         // std::locale, std::tolower
#include <algorithm>
# include <cmath>
# include "const.h"
# include <map>
# include <regex>
#include "cvector.h"
#include "VirtualHandler.h"
#include "OutputBuffer.h"
class ConfigFileHandler : public VirtualHandler {
	std::string _fileName;
	std::string _raw_data;

	std::map<std::string, OutputBuffer<float>*> _write_buffers;
public:
	
	std::map<std::string, std::string> paramsMap;
	std::map<std::string, std::string> filtersMap; // contains data of filter function
	std::map<std::string, std::string> filtersMapRaw; // contains data of filter with values and keys un lowered
	std::map<std::string, std::string> filtersKeysStat; // map of lowered to uppear keys
	ConfigFileHandler(const std::string& fileName);
	ConfigFileHandler();
	~ConfigFileHandler();

	void loadFile(const std::string& fileName);
	void processFile();
	void processData();
	int hasVariable(const std::string& variable_name);
	//double getDouble(const std::string& variable_name, const double& default_value = 0.0);
	//int getInt(const std::string& variable_name, const int& default_value = 0);
	//unsigned int getUnsignedInt(const std::string& variable_name, const unsigned int& default_value = 0);
	//long long getLong(const std::string& variable_name, const long long& default_value = 0);
	/**
	template<typename V> V getValue(const std::string& variable_name, const V& default_value = V(0)) {
		V result(default_value);
		if (hasVariable(variable_name)) result = parseToScalar<V>(getString(variable_name));
		return result;

	}

	*/
	//inline float getFloat(const std::string& variable_name) { return static_cast<float>(getDouble(variable_name)); }
	//std::vector<double> getDoubleVector(const std::string& variable_name);
	//std::vector<float> getFloatVector(const std::string& variable_name);
	//std::vector<int> getIntVector(const std::string& variable_name);
	std::string getString(const std::string& variable_name);
	template<typename V> typename std::enable_if<std::is_arithmetic<V>::value,V>::type getValue(const std::string& variable_name) {
		return parseToScalar<V>(getString(variable_name));
	}

	template<typename V> typename std::enable_if<std::is_same<V,std::string>::value, V>::type getValue(const std::string& variable_name) {
		return getString(variable_name);
	}

	template<typename V> typename std::enable_if<is_specialization<V, std::vector>::value, V>::type getValue(const std::string& variable_name) {
		typedef typename V::value_type ValueType;
		return parseToVector<ValueType>(getString(variable_name));
	}
	void setTargetForWrite(const std::string& write_target, void *target_object);
	void removeMinorFromMajorLocal(const std::string& minor, const std::string& major);
	void clearMajorBasic(const std::string& major, const int& terminate_data_object);
	void writeString(const std::string& minor, const std::string& major, const std::string& s) {}
	void write_vector(const std::string& minor, const std::string& major, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M);
	void write_vector(const std::string& minor, const std::string& major,float *v, const size_t& length, const size_t& offset, const size_t& M);
	inline int Is_Larget_Output(const std::string& major) { return 1; }
	void preAllocateMinor(const std::string& minor, const std::string& major, const size_t& M, const size_t& N) {}
	void Output_Major(const std::string& major, void **output_target) {}
	void Flush_Major(const std::string& major);
	inline int Is_Matlab_Formatted() { return 0; }
	int isAllocatedMemory(const std::string& minor, const std::string& major) { return isDefined(minor, major); }
	int getAllocatedMemory(const std::string& minor, const std::string& major) { return isDefined(minor, major); }
	void writeVectorString(const std::string& minor, const std::string& major, const std::vector<std::string>& v, const size_t& M) {}
	void write_map(const std::string& minor, const std::string& major,const mapper<std::string, double>& dataMap) {}
};

