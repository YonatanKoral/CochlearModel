#include "cochlea_utils.h"
std::string transformString(const std::string& input, std::function<char(char, const std::locale)> tr) {
	std::ostringstream oss;
	oss.str("");
	std::locale loc;

	for (auto elem : input)
		oss << tr(elem, loc);
	return oss.str();
}

std::string getFileType(const std::string& fileName) {
	return transformString(fileName.substr(fileName.find_last_of(".") + 1), std::toupper<char>);
}

std::string converToLower(const std::string& input) {
	std::ostringstream oss;
	oss.str("");
	std::locale loc;

	for (auto elem : input)
		oss << std::tolower(elem, loc);
	return oss.str();
}

std::string converToUpper(const std::string& input) {
	std::ostringstream oss;
	oss.str("");
	std::locale loc;

	for (auto elem : input)
		oss << std::toupper(elem, loc);
	return oss.str();
}


std::vector<std::string> splitToVector(const std::string& str, const std::string& regextest) {
	std::vector<std::string> vs;
	std::regex params_searcher(regextest);
	std::sregex_iterator rit(str.begin(), str.end(), params_searcher);
	std::sregex_iterator rend;
	while (rit != rend) {
		//cout << "detetcted: " << rit->str(1) << endl;
		vs.push_back(rit->str(1));
		++rit;
	}
	return vs;
}



std::vector<const char*> createCharMatrix(const std::vector<std::string>& v) {
	std::vector<const char*> cstrings;

	for (size_t i = 0; i < v.size(); ++i)
		cstrings.push_back(v[i].c_str());
	return cstrings;
}