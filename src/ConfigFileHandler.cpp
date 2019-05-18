#include "ConfigFileHandler.h"
using namespace std;

ConfigFileHandler::ConfigFileHandler(const string& fileName) : _fileName(fileName) {
	setDataStat(4);
	loadFile(_fileName);
}

ConfigFileHandler::ConfigFileHandler() {


}


ConfigFileHandler::~ConfigFileHandler() {
}


void ConfigFileHandler::loadFile(const string& fileName) {
	ifstream t(fileName);
	if (t.is_open()) {
		// read text file into raw string for processing
		_raw_data.assign((istreambuf_iterator<char>(t)),
			istreambuf_iterator<char>());
		processFile();
	} else {
		setDataStat(1);
	}
}

void ConfigFileHandler::processFile() {
	regex keyValue("([_[:alnum:]]+)\\s*=\\s*(\\\"(.*)\\\"|([^\\\"\\r\\n]+))$");
	const sregex_iterator end_iterator;
	auto it = sregex_iterator(_raw_data.begin(), _raw_data.end(), keyValue);
	//cout << "Processing " << _fileName << endl << "=================" << endl;
	for (sregex_iterator i = it; i != end_iterator; ++i) {
		std::smatch match = *i;
		//cout << "found keyValue: " << i->str() << endl;
		//cout << "key: " << i->str(1) << endl;
		//cout << "encapsulated value: " << i->str(2);
		//for (size_t si = 2; si < i->size(); si++) {
			//if (!i->str(2).empty()) {
				//cout << "match["<<si<<"] =  {" << i->str(si) << "}" << endl;
			//}
		//}
		string in_parameter(i->str(1));
		string in_value("");
		if (!i->str(3).empty()) {
			in_value.append(i->str(3));
			//cout << "decapsulated value: {" << i->str(3) << "}" << endl;
		} else if (!i->str(4).empty()) {
			in_value.append(i->str(4));
			//cout << "isolated value: {" << i->str(4) << "}" << endl;
		}
		paramsMap.insert(pair<string, string>(string(in_parameter), string(in_value)));
		//cout << "paramsMap[" << in_parameter << "] = " << paramsMap[in_parameter] << endl;
	}
	setDataStat(0);
}

void ConfigFileHandler::processData() { processFile();  }

int ConfigFileHandler::hasVariable(const std::string& variable_name) {
	//PrintFormat("Variable : %s parsed\n", variable_name.c_str());
	return static_cast<int>(paramsMap.count(variable_name));
}
string ConfigFileHandler::getString(const std::string& variable_name) {
	string result("");
	if (hasVariable(variable_name)) result.append(paramsMap[variable_name]);
	return result;
}
/**
double ConfigFileHandler::getDouble(const std::string& variable_name, const double& default_value) {
	double result(default_value);
	if (hasVariable(variable_name)) result = parseToScalar<double>(getString(variable_name));
	return result;
}
int ConfigFileHandler::getInt(const std::string& variable_name, const int& default_value) {
	int result(default_value);
	if (hasVariable(variable_name)) result = parseToScalar<int>(getString(variable_name));
	return result;
}
unsigned int ConfigFileHandler::getUnsignedInt(const std::string& variable_name, const unsigned int& default_value) {
	unsigned int result(default_value);
	if (hasVariable(variable_name)) result = parseToScalar<unsigned int>(getString(variable_name));
	return result;
}
long long ConfigFileHandler::getLong(const std::string& variable_name, const long long& default_value) {
	long long result(default_value);
	if (hasVariable(variable_name)) result = parseToScalar<long long>(getString(variable_name));
	return result;
}
//inline float getFloat(const std::string& variable_name) { return static_cast<float>(getDouble(variable_name)); }
vector<double> ConfigFileHandler::getDoubleVector(const std::string& variable_name) {
	vector<double> result(1,0.0);
	if (hasVariable(variable_name)) result = parseToVector<double>(getString(variable_name));
	return result;
}
vector<float> ConfigFileHandler::getFloatVector(const std::string& variable_name) {
	vector<float> result(1, 0.0f);
	if (hasVariable(variable_name)) result = parseToVector<float>(getString(variable_name));
	return result;
}
vector<int> ConfigFileHandler::getIntVector(const std::string& variable_name) {
	vector<int> result(1, 0);
	if (hasVariable(variable_name)) result = parseToVector<int>(getString(variable_name));
	return result;

}
*/
void ConfigFileHandler::setTargetForWrite(const std::string& write_target, void *target_object) {
	//std::cout << "targeting for write for file " << write_target << std::endl;
	_write_buffers[write_target] = (OutputBuffer<float>*)target_object;
}
void ConfigFileHandler::removeMinorFromMajorLocal(const std::string& minor, const std::string& major) {}
void ConfigFileHandler::clearMajorBasic(const std::string& major, const int& terminate_data_object) {
	auto it = _write_buffers.find(major);
	if (terminate_data_object) { 
		it->second->close_file();
		delete it->second; 
	}
	if (it != _write_buffers.end()) _write_buffers.erase(it);
}
void ConfigFileHandler::write_vector(const std::string& minor, const std::string& major,float *v, const size_t& length, const size_t& offset, const size_t& M) {
	//std::cout << __FILE__ << ",minor: " << minor << ",major: " << major << ",length: " << length << ",offset: " << offset << ",M: " << M << std::endl;
	_write_buffers[major]->append_buffer(v, static_cast<int>(length), static_cast<int>(offset));
}
void ConfigFileHandler::write_vector(const std::string& minor, const std::string& major, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M) {
	//std::cout << __FILE__ << ",minor: " << minor << ",major: " << major << ",length: " << length << ",offset: " << offset << ",M: " << M << std::endl;
	_write_buffers[major]->append_buffer(v, static_cast<int>(length), static_cast<int>(offset));
}

void ConfigFileHandler::Flush_Major(const std::string& major) {
	//std::cout << "FLUSH MAJOR: "<< __FILE__ << " : " << __LINE__ << std::endl;
	_write_buffers[major]->flush_buffer();
}