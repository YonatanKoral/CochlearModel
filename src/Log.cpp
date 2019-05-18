#include "Log.h" 


Log::Log() {
	oss.str("");
	interrupts = std::vector<std::chrono::high_resolution_clock::time_point>(INTERRUPTS_NUMBER);
	intmax_t imt = std::chrono::high_resolution_clock::period::den;
	setFlagsPerRound(64);
}


Log::~Log() {
}
Log& Log::operator=(const Log& l) {
	oss<<l.oss.str();
	std::copy(l.interrupts.begin(), l.interrupts.end(), std::back_inserter(interrupts));
	std::copy(l.stopers_all_rounds.begin(), l.stopers_all_rounds.end(), std::back_inserter(stopers_all_rounds));
	round_of_logging = l.round_of_logging;
	_flags_per_round = l._flags_per_round;
	return *this;
}
void Log::addRaw(const std::string& str) {
	oss << str << endl;
}
void Log::Print_For_Condition(const std::vector<std::string>& vs, const std::string& test, const std::vector<double>& source, const std::vector<float> target) {
	if (checkVector<std::string>(vs, test)) {
		addRaw(verboseVectors<double,float>(test, 4, 8, source, target));
		//cout << verboseVectors<double, float>(test, 4, 8, source, target) << endl;
	}
}

void Log::Print_Vector(const std::vector<float>& vf, const std::string& prefix) {
	oss << prefix << ": " << viewVector<float>(vf) << endl;
}


void Log::Print_Vector_For_Condition(const std::vector<std::string>& tester, const std::string& test, const std::vector<float>& vf, const std::string& prefix) {
	if (checkVector<std::string>(tester, test)) {
		Print_Vector(vf, prefix);
	}
}
long long Log::getElapsedTime(const std::chrono::high_resolution_clock::time_point& start_run, const std::chrono::high_resolution_clock::time_point& end_run) {
	return chrono::duration_cast<chrono::milliseconds>(end_run - start_run).count();
}
long long Log::getElapsedTime(const int& start_run, const int& end_run) {
	return getElapsedTime(interrupts[start_run], interrupts[end_run]);
}
void Log::elapsedTime(const std::string& prefix, const std::chrono::high_resolution_clock::time_point& start_run, const std::chrono::high_resolution_clock::time_point& end_run) {
	auto passed_msec = getElapsedTime(start_run,end_run);
	oss << "AudioLab "<< prefix << " time: " << passed_msec << " (msec)" << endl;
}

void Log::elapsedTimeView(const std::string& prefix, const int& index_start_run, const int& index_end_run, const int& condition) {
	if (condition) {
		elapsedTime(prefix, interrupts[index_start_run], interrupts[index_end_run]);
	}
}

void Log::elapsedTimeView(const std::string& prefix, const int& index_start_run, const int& index_end_run) {
	elapsedTimeView(prefix, index_start_run, index_end_run,1);
}

void Log::markTime(const int& index) {
	interrupts[index] = std::chrono::high_resolution_clock::now();
}

void Log::elapsedTimeInterrupt(const std::string& prefix, const int& index_start_run, const int& index_end_run, const int& condition) {
	markTime(index_end_run);
	elapsedTimeView(prefix, index_start_run, index_end_run,condition);
	timeAtFlag(index_end_run, static_cast<double>(getElapsedTime(index_start_run, index_end_run)), condition);
}

void Log::elapsedTimeInterrupt(const std::string& prefix, const int& index_start_run, const int& index_end_run) {
	elapsedTimeInterrupt(prefix, index_start_run, index_end_run,1);
}

void Log::clearLog() {
	oss.str(std::string());
}

void Log::flushLog() {
	cout << oss.str();
	clearLog();
}