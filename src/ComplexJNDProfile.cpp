#include "ComplexJNDProfile.h"

string ComplexJNDProfile::viewCaptured(const double& minValueFound, const vector<double>& tested_values, const vector<double>& minimums_captured) {
	ostringstream oss("");
	oss << "For " << show(-1);
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(1);
	oss << "minimum value found: " << minValueFound << "dB" << endl;
	viewVectorModes vvm = viewVectorModes{  0,  std::ios_base::scientific,  1 };
	if ( !minimums_captured.empty()) {
		oss << "out of #" << minimums_captured.size() << " candidates: " << viewVector<double>(minimums_captured, viewVectorModes{ 0, std::ios_base::fixed, 1 }) << endl;
	}
	if (!tested_values.empty()) {
		oss << "Number of tested value: " << tested_values.size() << endl << viewVector<double>(tested_values, viewVectorModes{ 0, std::ios_base::fixed, 1 }) << endl;
	}
	return oss.str();
}

float ComplexJNDProfile::calculateMinValue(const vector<double>& rawJNDValues, const vector<int>& Failed_Signal_Indexes, float eps, bool view_parts) {
	indexLocatorPredicate<double> claim_indexes(_intervals);
	indexLocatorPredicate<int> claim_indexes_int(_intervals);
	vector<double> relevant_values = claim_indexes(rawJNDValues);// vector<double>(_intervals.size(), 0.0f);
	std::vector<int> failed_convergence = claim_indexes_int(Failed_Signal_Indexes);
	vector<double> minimums_captured;
	vector<double> failed_convergence_found;
	minimums_captured.reserve(_intervals.size());
	failed_convergence_found.reserve(_intervals.size());
	calculateMinValueWarning = 0;
	int captured = 0;
	if (relevant_values.size() > 0) {
		//for (int i = 0; i < static_cast<int>(_intervals.size()); i++) {
		//	int raw_position = _intervals[i];
		//	if (raw_position < static_cast<int>(rawJNDValues.size())) relevant_values[i] = rawJNDValues[raw_position];
		//}

		for (int i = 1; i < static_cast<int>(relevant_values.size()) - 1; i++) {
			if (relevant_values[i] < relevant_values[i - 1] && relevant_values[i] < relevant_values[i + 1]) {
				minimums_captured.push_back(relevant_values[i]);
				failed_convergence_found.push_back(failed_convergence[i]);
				captured++;
			}
		}
		if (captured == 0) {
			calculateMinValueWarning = 1;
			if (relevant_values[0] < relevant_values[1]) { 
				_minValueFound = static_cast<float>(relevant_values[0]); 
				Failed_Convergence_Warning = static_cast<int>(failed_convergence_found[0]);
			}
			else if (relevant_values[relevant_values.size() - 1] < relevant_values[relevant_values.size() - 2]) { 
				_minValueFound = static_cast<float>(relevant_values[relevant_values.size() - 1]);
				Failed_Convergence_Warning = static_cast<int>(failed_convergence_found[relevant_values.size() - 1]);
			}
		} else {
			_minValueFound = static_cast<float>(minimums_captured[captured - 1]);
			Failed_Convergence_Warning = static_cast<int>(failed_convergence_found[captured - 1]);
		}
	}
	if (view_parts) {
		cout << viewCaptured(static_cast<double>(_minValueFound), relevant_values, minimums_captured);
		/*
		cout << "For " << show(-1)
			<< "_minValueFound = " << _minValueFound<<"\n"
			<< "captured = " << captured << "\n"
			<< "relevant_values = " << viewVector<double>(relevant_values) << "\n"
			<< "minimums_captured = " << viewVector<double>(minimums_captured) << "\n";
			*/

	}
	return _minValueFound;
}

float ComplexJNDProfile::calculateGradientMinMaxValue(const vector<double>& rawJNDValues, const vector<int>& Failed_Signal_Indexes, float eps, bool view_parts) {
	indexLocatorPredicate<double> claim_indexes(_intervals);
	vector<double> relevant_values = claim_indexes(rawJNDValues);// vector<double>(_intervals.size(), 0.0f);
	auto test_greater = [eps](const double& lhs, const double& rhs) { return rhs - lhs > static_cast<double>(eps);  };
	bool warning_boundary = false;
	calculateGradientMinMaxValueWarning = 0;
	if (relevant_values.size() > 0) {
		auto minValueFound = std::adjacent_find(relevant_values.begin(), relevant_values.end(), test_greater);
		if (minValueFound != relevant_values.end()) {
			_minValueFound = static_cast<float>(*minValueFound);
		} else {
			_minValueFound = static_cast<float>(relevant_values[0]);
			calculateGradientMinMaxValueWarning = 1;
			warning_boundary = true;
		}
	}
	if (view_parts) {
		if (warning_boundary) {
			cout << "****	Can't find slope of " << eps <<" ****" << endl;
		}
		cout << viewCaptured(static_cast<double>(_minValueFound), relevant_values, vector<double>(0));

	}
	return _minValueFound;
}
void ComplexJNDProfile::setInterval(int position, int index) { 
	if (index < static_cast<int>(_intervals.size())) {
		_intervals[index] = position; 
	}
}
void ComplexJNDProfile::updateBaseValues(float frequency, float dBSPLNoise, int length, const std::string &signal_name) {
	_dBSPLNoise = dBSPLNoise;
	_frequency = frequency;

	_intervals = vector<int>(length, 0);
	if (signal_name.empty()) _signal_name.clear();
	else _signal_name = signal_name;
}
ComplexJNDProfile::ComplexJNDProfile(float frequency, float dBSPLNoise, int length, const std::string &signal_name) :
_dBSPLNoise(dBSPLNoise),
_frequency(frequency){
	_minValueFound = MIN_INF_POWER_LEVEL;
	_intervals = vector<int>(length, 0);
}
std::string ComplexJNDProfile::showLegend(const int& details) {
	ostringstream oss;
	oss.str("");
	oss.setf(oss.boolalpha);
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(0);
	if ((details & 48)) {
		if ((details & 16)) {
			oss << "Minimum ";
		} else if ((details & 32)) {
			oss << "Wanted ";
		}
	}
	if ((details & 2)) {
		oss << "AI ";
	} else if ((details & 4)) {
		oss << "Rate ";
	}
	if (_dBSPLNoise >= JND_Sat_Value_DB) {
		oss << "Noiseless";
	} else {
		oss << "Noise " << _dBSPLNoise << " dB";
	}
	

	if ((details & 8)) {
		oss << ", ";
		if (!_signal_name.empty()) {
			oss << "Signal: " << _signal_name;
		} else {
			oss << "Frequency:" << _frequency << " HZ";
		}
	}
	
	if ((details & 1)) {
		oss << ",Minimum Found = " << _minValueFound << " dB";
	}
	return oss.str();
}
string ComplexJNDProfile::show(int index) {
	ostringstream oss;
	oss.str("");
	oss.setf(oss.boolalpha);
	if (index >= 0) {
		oss << "JND Complex Configuration #" << index;
	} else {
		oss << "JND Complex Configuration";
		if (!_intervals.empty()) {
			int interval_number = _intervals[0] + 1;
			oss << " #" << interval_number;
		}
	}
	oss << std::endl;
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(0);
	oss << "dBSPLNoise = " << _dBSPLNoise << "\n";
	oss << "_intervals = " << viewVector<int>(_intervals) << "\n";
	oss.setf(std::ios::scientific, std::ios::floatfield);
	oss.precision(5);
	oss << "_minValueFound = " << _minValueFound << "\n";
	oss.setf(std::ios::fixed, std::ios::floatfield);
	oss.precision(0);
	oss << "_frequency = " << _frequency << "\n";
	return oss.str();
}

ComplexJNDProfile::ComplexJNDProfile() :
_dBSPLNoise(0.0f),
_frequency(0.0f),
_intervals(10000, 0) {
	_minValueFound = MIN_INF_POWER_LEVEL;
}


ComplexJNDProfile::~ComplexJNDProfile() {
}
