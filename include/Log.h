#pragma once

#ifndef __LOG_H
#define __LOG_H
#include "const.h"
#include "cvector.h"
#include "cochlea_utils.h"
#include <ctime>
#include <chrono>
#include <type_traits>
#include <algorithm>


class Log {
private:
	std::ostringstream oss;
	
	std::vector<std::chrono::high_resolution_clock::time_point> interrupts;
	std::vector<float> stopers_all_rounds;
	int round_of_logging;
	int _flags_per_round;
public:
	Log();
	Log& operator=(const Log& l);
	~Log();
	inline int getFlagsPerRound() const { return _flags_per_round;  }
	inline void setFlagsPerRound(int flags_per_round) { _flags_per_round = flags_per_round;  }
	inline void advanceLogRound() { ++round_of_logging; stopers_all_rounds.resize(_flags_per_round*round_of_logging, 0.0); }
	inline void resetLogRound() { round_of_logging = 0; std::fill(stopers_all_rounds.begin(), stopers_all_rounds.end(), 0.0); advanceLogRound(); }
	inline void timeAtFlag(int flag_position_in_round, double timer_value,const int &condition) { 
		if (condition) stopers_all_rounds[_flags_per_round*(round_of_logging - 1) + flag_position_in_round] += static_cast<float>(timer_value);  
		//printf("marked flag %d at round %d value %.4f\n", flag_position_in_round, round_of_logging, timer_value);
	}
	inline std::vector<float> getTimers() { return stopers_all_rounds;  }
	void addRaw(const std::string& str);
	void Print_For_Condition(const std::vector<std::string>& vs, const std::string& test, const std::vector<double>& source, const std::vector<float> target);
	void Print_Vector(const std::vector<float>& vf, const std::string& prefix);
	void Print_Vector_For_Condition(const std::vector<std::string>& tester, const std::string& test, const std::vector<float>& vf, const std::string& prefix);
	long long getElapsedTime(const std::chrono::high_resolution_clock::time_point& start_run, const std::chrono::high_resolution_clock::time_point& end_run);
	long long getElapsedTime(const int& start_run, const int& end_run);
	void elapsedTime(const std::string& prefix, const std::chrono::high_resolution_clock::time_point& start_run, const std::chrono::high_resolution_clock::time_point& end_run);
	void elapsedTimeView(const std::string& prefix, const int& index_start_run, const int& index_end_run);
	void elapsedTimeInterrupt(const std::string& prefix, const int& index_start_run, const int& index_end_run);
	void elapsedTimeView(const std::string& prefix, const int& index_start_run, const int& index_end_run,const int& condition);
	void elapsedTimeInterrupt(const std::string& prefix, const int& index_start_run, const int& index_end_run, const int& condition);
	void markTime(const int& index);
	inline void setMarkedValueAtIndex(const int& index, const float &value, const int &condition) { 
		if (condition) {
			if (index >= stopers_all_rounds.size()) stopers_all_rounds.resize(index + 1, 0.0f);
			stopers_all_rounds[index] = value;
		}
	}
	void clearLog();
	inline std::string readLog() { return oss.str(); }
	void flushLog();
};
#endif

