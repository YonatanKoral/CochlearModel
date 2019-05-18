#pragma once
#include <vector>
#include <algorithm>
#include <mexplus/mexplus.h>
#include <type_traits>
#include "VirtualHandler.h"
#include "matrix.h"
#include "cvector.h"
#include "cochlea_utils.h"
bool advancedmxIsEmpty(const mxArray *mxa);


#ifdef CUDA_MEX_PROJECT
#ifndef CUDA_MEXPLUS
#define CUDA_MEXPLUS
	// Define two template specializations.
	template <class T> mxArray *construct_from(const mapper<std::string, T>& dataMap) {
		// Write your conversion code. For example,
		mexplus::MxArray struct_array = mexplus::MxArray(mexplus::MxArray::Struct());
		for (auto msd : dataMap.getData()) {
			struct_array.set(msd.first, msd.second);
		}
		for (auto mss : dataMap.getNames()) {
			struct_array.set(mss.first, mss.second);
		}
		//struct_array.set("x", value.x);
		//struct_array.set("y", value.y);
		// And so on...
		return struct_array.release();
	}
#endif
#endif


/**
* this class is utility to handle matlab translations methods, currently disabled due to mexplus
struct mxHandler {
	mxArray *_data;
	size_t _N;
	size_t _M;
	mxHandler(mxArray *data) {
		_data = data;
	}
	mxHandler(const size_t& M, const size_t& N, const std::string& anchorFieldName, const std::string& anchorFieldValue, const std::vector<std::string>& fieldNames=std::vector<std::string>()) {
		std::vector<std::string> v(fieldNames);
		v.push_back(anchorFieldName);
		_data = generateStruct(M, N, fieldNames);
		_M = M;
		_N = N;
		for (int n = 0; n < _N; n++) {
			for (int m = 0; m < _M; m++) {
				writeStringToStruct(m, n, anchorFieldName, anchorFieldValue);
			}
		}
		
	}
	mxHandler(const std::map<std::string, double> &smd) {
		std::vector<std::string> fieldNames = getAllKeys<std::string,double>(smd);
		_data = generateStruct(1, 1, fieldNames);
		write_map(0, 0, smd);
	}
	mxHandler(const mapper<std::string, double> &dmd) : mxHandler(dmd._data) {
		
		std::vector<std::string> fieldNames;
		std::transform(sms.begin(), sms.end(), std::back_inserter(fieldNames), first(sms));
		_data = generateStruct(1, 1, fieldNames);
		write_map(0, 0, sms);

		
		write_map(0, 0, dmd._names);
	}
	size_t getPosition(const size_t& M, const size_t& N) {
		return (M*_N+N);
	}
	void write_map(const size_t& M, const size_t& N,const std::map<std::string, double> &smd);
	void write_map(const size_t& M, const size_t& N, const std::map<std::string, std::string> &sms);
	void addField(const std::string& fieldName);
	mxArray *generateStruct(const size_t& M, const size_t& N, const std::vector<std::string>& fieldNames);
	void writeStringToStruct( const size_t& M, const size_t& N, const std::string& fieldName, const std::string& fieldValue);
	void writeDoubleToStruct(const size_t& M, const size_t& N, const std::string& fieldName, const double& fieldValue);
	void writeHandlerToStruct(const size_t& M, const size_t& N, const std::string& fieldName, const mxHandler& fieldValue);
};
*/
class MEXHandler : public VirtualHandler {
protected:
	mexplus::MxArray _data;
	//mxArray *_data_dynamic = NULL;
	map<std::string, mxArray * > _output_buffers;
	const string _file_name;
	const bool _terminate_data; // if true delete data on termination
	map<std::string, std::size_t> minor_buffers_positions;
	inline std::string bufferFullKey(const std::string& minor, const std::string& major) {
		std::ostringstream oss("");
		oss << minor << "." << major;
		return oss.str();
	}
	inline size_t bufferPosition(const std::string& minorwithmajor) {
		auto it = minor_buffers_positions.find(minorwithmajor);
		size_t result = 0;
		if (it != minor_buffers_positions.end()) {
			result = it->second;
		}
		return result;
	}
	inline size_t bufferPosition(const std::string& minor, const std::string& major) {
		auto fullKey = bufferFullKey(minor, major);
		return bufferPosition(fullKey);
	}
	inline size_t updateBufferPosition(const std::string& minor, const std::string& major, size_t length) {
		auto fullKey = bufferFullKey(minor, major);
		auto bposition = bufferPosition(fullKey);
		minor_buffers_positions[fullKey] = bposition + length;
		return (bposition + length);
	}
	inline void clearBufferPosition(const std::string& minor, const std::string& major) {
		auto fullKey = bufferFullKey(minor, major);
		erase_if(minor_buffers_positions, [fullKey](const std::pair<std::string, std::size_t> &subject) { return subject.first == fullKey; });
	}
public:
	//MEXHandler(mxArray *data) : _data(data), _terminate_data(false), _file_name("") { data_found = 0; }
	MEXHandler(const mxArray *data) : _data(data), _terminate_data(false), _file_name("") {
		data_found = 0; 
	}
	MEXHandler(mxArray *data) : _data(data), _terminate_data(true), _file_name("") {
		//_data_dynamic = data;
		data_found = 0;
	}
	MEXHandler(const std::string& file_name);
	MEXHandler();
	void loadStruct(const std::string& file_name);
	void loadStruct();
	void processData();
	int hasVariable(const std::string& variable_name);
	inline const mxArray *getVariable(const std::string& variable_name) const { return _data.at(variable_name); }
	
	template<typename V> typename std::enable_if<is_specialization<V, std::vector>::value || std::is_same<V, std::string>::value || std::is_arithmetic<V>::value, V>::type getValue(const std::string& variable_name) {
		V v1 = _data.at<V>(variable_name);
		return v1;
	}
	//simplifiedData processMxArray(const std::string& variable_name);
	void loadStructFromMex(mxArray *input_struct);
	void updateMajor(std::string majorName, mxArray *mxa);
	void removeMinorFromMajorLocal(const std::string& minor, const std::string& major);
	void setTargetForWrite(const std::string& write_target, void *target_object);
	void clearMajorBasic(const std::string& major, const int& terminate_data_object);
	//inline int isValidData(const simplifiedData& sd) { return sd.length > 0 && sd.data != NULL; }
	inline int Is_Larget_Output(const std::string& major) { return 0; }
	void write_vector(const std::string& minor, const std::string& major, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M);
	void write_vector(const std::string& minor, const std::string& major, float *v, const size_t& length, const size_t& offset, const size_t& M);
	void Flush_Major(const std::string& major);
	//void preAllocateMinor(const std::string& minor, const std::string& major, const size_t& M, const size_t& N);
	void preAllocateMinor(const std::string& minor, const std::string& major, const size_t& M, const size_t& N) {}
	void writeString(const std::string& minor, const std::string& major, const std::string& s);
	void Output_Major(const std::string& major, void **output_target);
	void removeInputData();
	mexplus::MxArray claimFieldHandler(const std::string& minor, const std::string& major);
	//mxArray *claimField(const std::string& minor, const std::string& major);
	//mxArray *claimStruct(const std::string& minor, const std::string& major);
	int getAllocatedMemory(const std::string& minor, const std::string& major);
	int isAllocatedMemory(const std::string& minor, const std::string& major) { return isDefined(minor, major) && _output_buffers.count(major)>0 && getAllocatedMemory(minor,major)>0; }
	void writeVectorString(const std::string& minor, const std::string& major, const std::vector<std::string>& v, const size_t& M);
	void write_map(const std::string& minor, const std::string& major, const mapper<std::string, double>& dataMap);
	inline int Is_Matlab_Formatted() { return 1; }
	~MEXHandler();
};

