#pragma once
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <sstream>
# include <iostream>
#include <string>
#include <type_traits>
#include <fstream>
#include <streambuf>
#include <locale>         // std::locale, std::tolower
#include <algorithm>
# include <cmath>
# include "const.h"
# include <map>
# include <regex>
#include "cochlea_utils.h"
#include "cvector.h"
#include "Log.h"
class ConfigFileHandler;
#ifdef MATLAB_MEX_FILE
class MEXHandler;
#endif

class VirtualHandler;
template<typename V> V getValueStatic(VirtualHandler *vh,const std::string& variable_name) { 
	//std::cout << "run VirtualHandler getValue" << std::endl; 
	if (ConfigFileHandler *cfh = dynamic_cast<ConfigFileHandler*>(vh)) {
		//std::cout << "run ConfigFileHandler getValue" << std::endl;
		return cfh->getValue<V>(variable_name);
	}
#ifdef MATLAB_MEX_FILE
	else {
		if (MEXHandler *mexh = dynamic_cast<MEXHandler*>(vh)) {
			//std::cout << "run MEXHandler getValue" << std::endl;
			return mexh->getValue<V>(variable_name);
		}
	}
#endif
	return V(); 
}

/**
* this is pure virtual class that abstracts Input/Output from matlab and/or Binary files Harddrive
*/

class VirtualHandler {
protected:
	int data_found; // 0 - yes, 1 - file not found, 2 - variable not found, 3 - variable not structure, 4 - data not read yet
	std::map<std::string, std::string> minor_to_major_tags;
	std::map<std::string, std::vector<std::string> > major_to_minor_tags;
	std::string primary_major;
public:
	VirtualHandler();
	virtual ~VirtualHandler();
	virtual void processData() = 0;
	// write_target is major target write, tags to write are sub tags to write so major target can  contain multiple target
	virtual void setTargetForWrite(const std::string& write_target, void *target_object) = 0;
	virtual int hasVariable(const std::string& variable_name) =0;
	
	inline int dataNotRead() { return data_found == 4; }
	inline int dataNotStructure() { return data_found == 3; }
	inline int dataNoVariable() { return data_found == 2; }
	inline int dataNoFile() { return data_found == 1; }
	inline int dataOk() { return data_found == 0; }
	inline int getHandlerErrorCode() { return data_found;  }
	inline void setDataStat(const int& stat) { data_found = stat; }
	inline std::string getMajor(const std::string& minor) { return minor_to_major_tags[minor]; }
	inline int hasMajor(const std::string& minor) { return static_cast<int>(minor_to_major_tags.count(minor)); }
	inline int isMajor(const std::string& major) { return static_cast<int>(major_to_minor_tags.count(major)); }
	void setTargetsForWrite(const std::string& major_write_target, const std::vector<std::string>& minor_targets, void *target_object) {
		setTargetForWrite(major_write_target, target_object);
		for (auto minor : minor_targets) {
			addMinorToMajor(minor, major_write_target);
		}
	}
	virtual void removeMinorFromMajorLocal(const std::string& minor, const std::string& major) = 0;
	inline void removeMinorFromMajor(const std::string& minor) {
		if (hasMajor(minor)) {
			std::string major = getMajor(minor);
			removeMinorFromMajorLocal(minor, major);
			auto it = minor_to_major_tags.find(minor);
			if (it != minor_to_major_tags.end()) minor_to_major_tags.erase(it);
			std::remove_if(major_to_minor_tags[major].begin(), major_to_minor_tags[major].end(), [minor](std::string &subject) { return subject == minor; });
		}
	}
	inline void addMinorToMajor(const std::string& minor, const std::string& major) {
		removeMinorFromMajor(minor); // clear old minors
		if (!isMajor(major)) {
			std::vector<std::string> mv = { minor };
			major_to_minor_tags[major] = mv;
		} else {
			major_to_minor_tags[major].push_back(minor);
		}
		
		minor_to_major_tags[minor] = major;
	}
	// gets lists of all minor fields belong to major structure
	std::vector<std::string> getAllMinors(const std::string& major) {
		std::vector<std::string> result(major_to_minor_tags[major]);
		return result;
	}
	// clears major pointer without destroying the major object itself
	virtual void clearMajorBasic(const std::string& major, const int& terminate_data_object) = 0;
	void clearMajor(const std::string& major, const int& terminate_data_object) {
		clearMajorBasic(major, terminate_data_object);
		erase_if(major_to_minor_tags, [major](const std::pair<std::string, std::vector<std::string>  > &subject) { return subject.first == major; });
	}
	inline void removeMajor(const std::string& major, const int& terminate_data_object) {
		std::vector<std::string> minors = getAllMinors(major);
		for (auto minor : minors) {
			removeMinorFromMajor(minor);
		}
		clearMajor(major, terminate_data_object);
	}
	void clearPrimaryMajor(const int& terminate_data_object) {
		//std::cout << "clearing primary major... " << std::endl;
		if (hasPrimaryMajor()) {
			//std::cout << "clearing primary major... " << primary_major << std::endl;
			removeMajor(primary_major, terminate_data_object);
			primary_major.clear();
		}
	}
	void setPrimaryMajor(const std::string& major) {
		primary_major = major;
	}
	inline int hasPrimaryMajor() { return !primary_major.empty();  }
	// write string to major.minor
	virtual void writeString(const std::string& minor, const std::string& major, const std::string& s) = 0;
	void writeString(const std::string& minor, const std::string& s) {
		if (hasMajor(minor)) {
			writeString(minor, getMajor(minor), s);
		} else if (hasPrimaryMajor()) {
			writeString(minor, primary_major, s);
		}
	}
	// allocat field major.minor of floats
	virtual void preAllocateMinor(const std::string& minor, const std::string& major, const size_t& M, const size_t& N) = 0;
	int isDefined(const std::string& minor, const std::string& major) { return hasMajor(minor) && isMajor(major)&&hasMajor(minor) && getMajor(minor).compare(major) == 0; }
	virtual int isAllocatedMemory(const std::string& minor, const std::string& major) = 0;
	virtual int getAllocatedMemory(const std::string& minor, const std::string& major) = 0;
	virtual void writeVectorString(const std::string& minor, const std::string& major, const std::vector<std::string>& v, const size_t& M) = 0;
	void writeVectorString(const std::string& minor, const std::vector<std::string>& v, const size_t& M) {
		if (!v.empty()) {
			if (hasMajor(minor)) {
				//std::cout << "founds major..." << std::endl;
				writeVectorString(minor, getMajor(minor), v, M);
			} else if (hasPrimaryMajor()) {
				writeVectorString(minor, primary_major, v, M);
			}
		}
	}
	// write vector for sub field major.minor in output
	virtual void write_vector(const std::string& minor, const std::string& major, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M) = 0;
	virtual void write_vector(const std::string& minor, const std::string& major,float *v, const size_t& length, const size_t& offset, const size_t& M) = 0;
	void write_vector(const std::string& minor, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M) {
		//std::cout << __FUNCTION__ << ",minor: " << minor << ",has major? " << (hasMajor(minor) ? getMajor(minor) : "No") << ",length: " << length << ",offset: " << offset << ",M: " << M << std::endl;
		if (hasMajor(minor)) {
			//std::cout << "founds major..." << std::endl;
			write_vector(minor, getMajor(minor), v,length,offset,M);
		} else if (hasPrimaryMajor()) {
			write_vector(minor, primary_major, v, length, offset, M);
		}
	}
	void write_vector(const std::string& minor,float *v, const size_t& length, const size_t& offset, const size_t& M) {
		//std::cout << __FUNCTION__ << ",minor: " << minor << ",has major? " << (hasMajor(minor)?getMajor(minor):"No") << ",length: " << length << ",offset: " << offset << ",M: " << M << std::endl;
		if (hasMajor(minor)) {
			//std::cout << "founds major..." << std::endl;
			write_vector(minor, getMajor(minor), v, length, offset, M);
		} else if (hasPrimaryMajor()) {
			write_vector(minor, primary_major, v, length, offset, M);
		}
	}
	virtual void write_map(const std::string& minor, const std::string& major,const mapper<std::string,double>& dataMap) = 0;
	void write_map(const std::string& minor,const mapper<std::string, double>& dataMap) {
		if (hasMajor(minor)) {
			//std::cout << "founds major..." << std::endl;
			write_map(minor, getMajor(minor), dataMap);
		} else if (hasPrimaryMajor()) {
			write_map(minor, primary_major, dataMap);
		}
	}


	void VirtualHandler::flushToIOHandler(Log &log, const std::string& minor_tag) {
		std::string ostring = log.readLog();
		if (!ostring.empty()) {
			auto spllines = splitToVector(ostring, "([^\n]+)");
			//std::cout << "ostring : " << ostring << std::endl;
			//std::cout << "spllines : " << viewVector<std::string>(spllines,1) << std::endl;
			if (spllines.size() > 0) {
				writeVectorString(minor_tag, spllines, 0);
			}// else {
			 //std::cout << "log(name='" << minor_tag << "')<" << ostring << ">" << std::endl;
			 ///}
		}
		log.clearLog();
	}
	virtual void Flush_Major(const std::string& major) = 0;
	virtual void Output_Major(const std::string& major, void **output_target) = 0;
	// is true for config file handler, false for mex handler due to break in inline
	virtual int Is_Larget_Output(const std::string& major) = 0;
	virtual int Is_Matlab_Formatted() = 0;
	// SFINAE get vector/scalar/string value
	template<typename T> typename std::enable_if<is_specialization<T, std::vector>::value || std::is_arithmetic<T>::value || std::is_same<T, std::string>::value, T>::type getValue(const std::string& variable_name) {
		T v1 =  getValueStatic<T>(this, variable_name);
		return v1;
	}
};

