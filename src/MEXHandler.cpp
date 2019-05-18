#include "MEXHandler.h"
#include "mex.h"
#include "mat.h"
#include "cvector.h"
using namespace std;

bool advancedmxIsEmpty(const mxArray *mxa) {
	return (mxa == NULL || mxIsEmpty(mxa));
}
/**
mxArray *mxHandler::generateStruct(const size_t& M, const size_t& N,const std::vector<std::string>& fieldNames) {
	std::vector<std::string> minor_outputs2 = std::vector<std::string>();
	if (!fieldNames.empty()) {
		std::copy(fieldNames.begin(), fieldNames.end(), std::back_inserter(minor_outputs2));
	}
	auto char_matrix = createCharMatrix(minor_outputs2);
	auto output_struct = mxCreateStructMatrix(M, N, static_cast<int>(char_matrix.size()), char_matrix.data());
	return output_struct;
}
void mxHandler::writeStringToStruct(const size_t& M, const size_t& N, const std::string& fieldName, const std::string& fieldValue) {
	addField(fieldName);
	char *output_buf = (char *)mxCalloc(fieldValue.size() + 1, sizeof(char));
	std::copy(fieldValue.data(), fieldValue.data() + fieldValue.size(), stdext::checked_array_iterator<char*>(output_buf, fieldValue.size()));
	auto mxm = mxCreateString(output_buf);
	mxSetField(_data, getPosition(M, N), fieldName.c_str(), mxm);
}

void mxHandler::writeHandlerToStruct(const size_t& M, const size_t& N, const std::string& fieldName, const mxHandler& fieldValue) {
	addField(fieldName);
	mxSetField(_data, getPosition(M, N), fieldName.c_str(), fieldValue._data);

}
void mxHandler::writeDoubleToStruct(const size_t& M, const size_t& N, const std::string& fieldName, const double& fieldValue) {
	addField(fieldName);
	auto mxd = mxCreateDoubleMatrix(1, 1, mxREAL);
	*(mxGetPr(mxd)) = fieldValue;
	mxSetField(_data, getPosition(M,N), fieldName.c_str(), mxd);
}
void mxHandler::write_map(const size_t& M, const size_t& N,const std::map<std::string, double> &smd) {
	for (const auto& elem : smd) {
		writeDoubleToStruct(M, N, elem.first, elem.second);
	}
}
void mxHandler::write_map(const size_t& M, const size_t& N, const std::map<std::string, std::string> &sms) {
	for (const auto& elem : sms) {
		writeStringToStruct(M, N, elem.first, elem.second);
	}
}
void mxHandler::addField( const std::string& fieldName) {
	if (mxGetFieldNumber(_data, fieldName.c_str()) == -1) {
		if (mxAddField(_data, fieldName.c_str()) == -1) {
			ostringstream oss("");
			oss << "Error: could not add field " << fieldName;
			throw std::runtime_error(oss.str());
		}
	}

}
*/


MEXHandler::MEXHandler(const string& file_name) :_terminate_data(true), _file_name(file_name) {
	
}
// for output scheme
MEXHandler::MEXHandler() : _terminate_data(false) {

}
void MEXHandler::loadStruct() {
	loadStruct(_file_name);
}
void MEXHandler::processData() {
	loadStruct(_file_name);
}
void MEXHandler::loadStruct(const std::string& file_name) {
	setDataStat(4);
	//std::cout << "loadStruct, file opens... " << file_name<<std::endl;
	MATFile *mtf = matOpen(file_name.c_str(), "r");
	//std::cout << "loadStruct, file opens passed... " << file_name << std::endl;
	const char *fname;
	if (mtf == NULL) {
		//std::cout << "loadStruct, file not opened... " << file_name << std::endl;
		setDataStat(1);
		return;
	}
	//std::cout << "loadStruct, file opened successfully..., loading variable " << file_name << std::endl;
	_data = mexplus::MxArray(matGetNextVariable(mtf, &fname));
	matClose(mtf);
	//std::cout << "loadStruct, variable loaded... "<< std::endl;
	if ( !_data) {
		setDataStat(2);
		return;
	}
	//std::cout << "loadStruct, variable loaded... " << fname << std::endl;
	//_data = const_cast<mxArray *>(_data_dynamic);
	if ( _data.isStruct() ) {
		setDataStat(0);
		//std::cout << "loadStruct, variable " << fname << " is struct" << std::endl;
	} else {
		//std::cout << "loadStruct, variable " << fname << " is " << mxGetClassName(_data)  <<", aborts..."<< std::endl;
		setDataStat(3);
		return;
	}
	
}
void MEXHandler::removeInputData() {
	//std::cout << "removeInputData start...." << std::endl;
	if (_terminate_data &&  _data.isOwner() ) {
		//std::cout << "removeInputData ready to destory...." << std::endl;
		_data.reset();
	}
	//std::cout << "removeInputData end...." << std::endl;
}
MEXHandler::~MEXHandler() {
	removeInputData();
}



void MEXHandler::Output_Major(const std::string& major, void **output_target) {
	if (hasMajor(major)) {
		mxArray **mxa = (mxArray **)output_target;
		*mxa = mxDuplicateArray(_output_buffers[major]);
		//*mxa = _output_buffers[major].clone();
	}
}

void MEXHandler::setTargetForWrite(const std::string& write_target, void *target_object) {
	//mxArray *mxa = (mxArray *)target_object;
	mxArray *mxa = (mxArray *)target_object;
	_output_buffers[write_target] = mxa; //std::shared_ptr<mxArray>(mxa);
}
void MEXHandler::clearMajorBasic(const std::string& major, const int& terminate_data_object) {
	auto it = _output_buffers.find(major);
	if (it != _output_buffers.end()) {
		if (terminate_data_object) {
			//std::cout << "MEXHandler::clearMajor(" << major <<","<<terminate_data_object<<")" << std::endl;
			mxDestroyArray(it->second);
		}
		_output_buffers.erase(it);
	}
}


// for internal use only updates major after release
void MEXHandler::updateMajor(std::string majorName, mxArray *mxa) {
	_output_buffers[majorName] = mxa; //std::shared_ptr<mxArray>(mxa);
}
void MEXHandler::write_vector(const std::string& minor, const std::string& major, const std::vector<float>& v, const size_t& length, const size_t& offset, const size_t& M) {
	float *fdata = (float *)v.data();
	write_vector(minor, major, fdata, length, offset, M);
}
void MEXHandler::write_vector(const std::string& minor, const std::string& major, float *v, const size_t& length, const size_t& offset,const size_t& M) {
	//std::cout << __FILE__ << ": (minor=" << minor << ",major=" << major << ",length= " << length << ",offset= "<<offset<<",M="<<M<<")" << std::endl;
	auto mxa = claimFieldHandler(minor, major);
	/**
	mxArray *mxm = mxGetField(mxa, 0, minor.c_str());
	size_t N = static_cast<size_t>(length) / M;
	if (advancedmxIsEmpty(mxm)) {
		//std::cout << minor << " allocated in "<< major <<"length: " <<length << std::endl;
		preAllocateMinor(minor, major, M, length / M);
		mxm = mxGetField(mxa, 0, minor.c_str());
		//mxm = mxCreateNumericMatrix(M, N, mxSINGLE_CLASS, mxREAL);
		//mxSetField(mxa, 0, minor.c_str(), mxm);
	} 
	// now check there is enough memory ar re allocate to conpensate
	size_t new_position = updateBufferPosition(minor, major, length);
	size_t old_position = new_position - static_cast<size_t>(length);
	size_t mwM1 = mxGetM(mxm);
	size_t mwN1 = mxGetN(mxm);
	// not enough cells left, lets add some first
	size_t Nleft = mwN1 - (old_position / M);

	//std::cout << minor << " dimensions: (M= " << M << ",N= " << N << ",NLeft=" << Nleft << ",offset=" << offset << ",length="<<length<<")" << std::endl;
	if (Nleft < N) {
		preAllocateMinor(minor, major, M, N - Nleft);
	}
	//std::cout << "data allocated successfully for " << minor << std::endl;
	float *output_data = (float *)mxGetData(mxm);
		
	float *output_data_start = output_data + old_position;
	//float *output_data_end = output_data_start + length + 1;
	*/
	size_t Mm1 = M;
	if (Mm1 == 0) Mm1 = 1;
	//PrintFormat("outputs: %s length %d,offset:%d,Mm1=%d\n", minor.c_str(), length, offset, Mm1);
	mxa.append<float>(minor, v, length, offset, 0, Mm1);
	updateMajor(major, mxa.release());
	/**
	std::vector<float> ovdata;
	if (mxa.at(minor)) {
		ovdata = mxa.at< std::vector<float> >(minor);
	}
	auto vstart = std::next(v, offset);
	auto vend = std::next(vstart, length+1);
	//std::copy(vstart, vend, stdext::checked_array_iterator<float *>(output_data_start,length+1-offset));
	std::copy(vstart, vend, std::back_inserter(ovdata));
	auto mxm = mexplus::MxArray::from(ovdata);
	mxSetM(mxm, M);
	mxSetN(mxm, ovdata.size()/M);
	mxa.set(minor, mxm);
	mxa.release();
	*/
}

/**
// can be used for additional allocation
void MEXHandler::preAllocateMinor(const std::string& minor, const std::string& major, const size_t& M, const size_t& N) {
	//std::cout << __FILE__ << ",preAllocateMinor: (minor=" << minor << ",major=" << major << ",M= " << M << ",N= " << N << ")" << std::endl;
	auto mxa = claimFieldHandler(minor, major);
	
	mxArray *mxm = mxGetField(mxa, 0, minor.c_str());
	if (advancedmxIsEmpty(mxm)) {
		//mxm = mxCreateNumericMatrix(M, N, mxSINGLE_CLASS, mxREAL);
		//mxm = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL); // Create an empty array 
		mxSetM(mxm, M); // Set the dimensions to M x N 
		mxSetN(mxm,N);
		mxSetData(mxm, mxMalloc(sizeof(float)*M*N)); // Allocate memory for the array
		mxSetField(mxa, 0, minor.c_str(), mxm);
	} else {
		size_t oldM = mxGetM(mxm);
		size_t oldN = mxGetN(mxm);
		if (M != oldM) {
			std::cout << "re allocate " << bufferFullKey(minor, major) << " M dimension mismatch between old value = " << oldM << " and new_value = " << M << std::endl;
			//M = std::max(oldM, M);
			//mxSetM(mxm, M);
		}
		mxSetN(mxm, N + oldN);
		float *ptr = (float *)mxGetData(mxm);
		float *newptr = (float *)mxRealloc(ptr, M*(N+oldN)*sizeof(*ptr));
		mxSetData(mxm, (void *)newptr);

	}
}
*/

int MEXHandler::getAllocatedMemory(const std::string& minor, const std::string& major) {
	int nodes = 0;
	auto mxa = claimFieldHandler(minor, major);
	if (mxa ) {
		auto mxm = mexplus::MxArray(mxa.at(minor));
		if ( mxm ) nodes = static_cast<int>(mxm.size());
	}
	updateMajor(major, mxa.release());
	return nodes;
}

void MEXHandler::writeVectorString(const std::string& minor, const std::string& major, const std::vector<std::string>& v,const size_t& M) {
	auto mxa = mexplus::MxArray(claimFieldHandler(minor,major));
	
	//std::vector<std::string> prev;
	/**
	if (mxa && mxa.at(minor) ) {
		mxa.at(minor);
	}
	*/
	//std::copy(v.begin(), v.end(), std::back_inserter(prev));
	
	size_t Mm1 = M;
	if (Mm1 == 0) {
		Mm1 = 1;
	}
	size_t N = v.size() / Mm1;
	if (Mm1*N != v.size()) {
		printf("M=%d,N=%d,minor=%s,size()=%d\n",Mm1,N,minor.c_str(),v.size());
		//throw std::runtime_error("M*N not aligned to v size");
	}
	auto mxfield = mexplus::MxArray(mexplus::MxArray::Cell(static_cast<int>(Mm1), static_cast<int>(N)));
	for (size_t i = 0; i < v.size(); i++) {
		mxfield.set(i, v[i]);
	}
	//mxSetM(mxfield, Mm1);
	//mxSetN(mxfield, N);
	mxa.set(minor, mxfield.release());
	updateMajor(major, mxa.release());
	/**
	size_t Min = M;
	if (Min == 0) Min = v.size();
	auto nodes = getAllocatedMemory(minor, major);
	if (nodes == 0) {
		//auto mxm = mxCreateCharMatrixFromStrings(v.size(), createCharMatrix(v).data());
		auto mxm = mxCreateCellMatrix(Min,v.size()/Min);
		//std::cout << minor << " mxCreateCellMatrix("<<Min<<","<<(v.size()/Min)<<")" << std::endl;
		size_t i = 0;
		for (auto vs : v) {
			//std::cout << minor<<".cell(" << i << ")=" << v[i].c_str() <<"///"<<v[i]<< std::endl;
			auto strcell = mxCreateString(v[i].c_str());
			mxSetCell(mxm, i, strcell);
			i++;
		}
		mxSetField(mxa, 0, minor.c_str(), mxm);
	} else {
		auto ptr = mxGetField(mxa, 0, minor.c_str());
		mwIndex MN[2];
		MN[1] = mxGetN(ptr);
		MN[0] = mxGetM(ptr);
		size_t M1index = MN[0];
		size_t M2index = MN[0];
		MN[0] += v.size()/Min;
		//std::cout << minor << " extended mxCreateCellMatrix("<<MN[0]<<","<<MN[1]<<")" << std::endl;
		if ( mxSetDimensions(ptr, MN, 2) == 0) {
			for (auto vs : v) {
				//std::cout << minor << ".cell(" << M1index << ")=" << v[M1index - M2index].c_str() << "///" << v[M1index - M2index] << std::endl;
				mxSetCell(ptr, M1index, mxCreateString(v[M1index - M2index].c_str()));
				M1index++;
			}
		}
	}
	*/
}
mexplus::MxArray MEXHandler::claimFieldHandler(const std::string& minor, const std::string& major) {
	if (!isDefined(minor, major)) {
		// minor,major combination is missing, lets add it
		addMinorToMajor(minor, major);
	}
	auto it = _output_buffers.find(major);
	if (it != _output_buffers.end()) {
		//mexplus::MxArray mxa = it->second;
		
		return mexplus::MxArray(_output_buffers[major]);
	}
	return mexplus::MxArray();
}


/**
mxArray *MEXHandler::claimField(const std::string& minor, const std::string& major) {
	auto mxa = mexplus::MxArray(claimFieldHandler(minor, major));
	auto mxm = mxa.at(minor);
	mxa.release();
	return mxm;
}
*/
/**
mxArray *MEXHandler::claimStruct(const std::string& minor, const std::string& major) {
	auto mxm = claimField(minor, major);
	if (advancedmxIsEmpty(mxm)) {
		auto mxa = claimFieldHandler(minor, major);
		mxSetField(mxa, 0, minor.c_str(), mxHandler(1, 1, "sub_struct", minor)._data);
		mxm = claimField(minor, major);
	}
	if (!mxIsStruct(mxm)) {
		std::ostringstream oss("");
		oss << major << "." << minor << " is not structure";
		throw std::runtime_error(oss.str());
	}
	return mxm;
}
*/

void MEXHandler::write_map(const std::string& minor, const std::string& major,const mapper<std::string, double>& dataMap) {
	auto mxa = mexplus::MxArray(claimFieldHandler(minor, major));
	mxa.set(minor, construct_from<double>(dataMap));
	updateMajor(major, mxa.release());
	//auto structedMap = mxHandler(dataMap)._data;
	//mxSetField(mxa, 0, minor.c_str(), structedMap);
}
void MEXHandler::writeString(const std::string& minor, const std::string& major, const std::string& s) {
	auto mxa = mexplus::MxArray(claimFieldHandler(minor, major));
	mxa.set(minor, s);
	updateMajor(major, mxa.release());
	/**
	auto mxm = claimField(minor, major);
	if (advancedmxIsEmpty(mxm)) {
		char *output_buf = (char *)mxCalloc(s.size() + 1, sizeof(char));
		std::copy(s.data(), s.data() + s.size(), stdext::checked_array_iterator<char*>(output_buf, s.size()));
		mxm = mxCreateString(output_buf);
		mxSetField(mxa, 0, minor.c_str(), mxm);
	} else {
		char *buf = (char *)mxGetData(mxm);
		if (sizeof(buf) < (s.size() + 1)*sizeof(char)) {
			auto second_buffer = (char *)mxRealloc(buf, (s.size() + 1)*sizeof(char));
			buf = second_buffer;
		}
		std::copy(s.data(), s.data() + s.size(), stdext::checked_array_iterator<char*>(buf, s.size()));
		mxSetData(mxm, (void *)buf);
	}
		//if (mxm == NULL) {
	
		//}
		*/
}
int MEXHandler::hasVariable(const string& variable_name) {
	//std::cout << std::boolalpha << "hasVariable, testing " << variable_name << ",found? " << (getVariable(variable_name) != NULL) << std::endl;
	return getVariable(variable_name) != NULL;
}

// get numerical scalar
/**
double MEXHandler::getDouble(const std::string& variable_name, const double& default_value) {
	//double result(default_value);
	//if (hasVariable(variable_name)) result = mxGetScalar(getVariable(variable_name));
	double result = _data.at< double >(variable_name);
	//std::cout << "getDouble " << variable_name << " : " << result << std::endl;
	return result;
}
*/

/**
int MEXHandler::getInt(const std::string& variable_name, const int& default_value) {
	//int result(0);
	//int result = static_cast<int>(getDouble(variable_name,static_cast<double>(default_value)));
	int result = _data.at< int >(variable_name, default_value);
	//std::cout << "getInt " << variable_name << " : " << result << std::endl;
	return result;
}
*/

/**
unsigned int MEXHandler::getUnsignedInt(const std::string& variable_name, const unsigned int& default_value) {
	//unsigned int result = static_cast<int>(getDouble(variable_name, static_cast<double>(default_value)));
	unsigned int result = _data.at< unsigned int >(variable_name,default_value);
	//std::cout << "getUnsignedInt " << variable_name << " : " << result << std::endl;
	return result;
}

long long MEXHandler::getLong(const std::string& variable_name, const long long& default_value) {
	//long long result = static_cast<long long>(getDouble(variable_name, static_cast<double>(default_value)));
	long long result = _data.at< long long >(variable_name, default_value);
	return result;
}
*/

void MEXHandler::removeMinorFromMajorLocal(const std::string& minor, const std::string& major) {
	clearBufferPosition(minor, major);
}

//std::string MEXHandler::getString(const std::string& variable_name) {
//	std::string result = _data.at<std::string>(variable_name);
	/**
	if (hasVariable(variable_name)) {
		mxArray *mxa = getVariable(variable_name);
		if (mxIsChar(mxa)) {
			size_t buflen = mxGetNumberOfElements(mxa) + 1;
			char *buf = (char *)malloc(buflen*sizeof(char));
			if (mxGetString(mxa, buf, buflen) == 0) {
				//std::cout << "char["<<buflen<<"] pointer to " << buf << std::endl;
				result.append(buf, buflen-1);
			}


			free(buf);
		}
	}
	*/
	//std::cout << "getString " << variable_name << " : " << result << std::endl;
	//return result;
//}

/**
simplifiedData MEXHandler::processMxArray(const string& variable_name) {
	simplifiedData sd;
	sd.length = 0;
	if (hasVariable(variable_name)) {
		mxArray *mxa = getVariable(variable_name);
		const mwSize *N = mxGetDimensions(mxa);
		mwSize numDimensions = mxGetNumberOfDimensions(mxa);
		sd.length = 1;
		for (mwSize i = 0; i < numDimensions; i++) {
			sd.length = sd.length*N[i];
		}
		sd.data = mxGetData(mxa);
	}
	return sd;
}
*/
/**
vector<double> MEXHandler::getDoubleVector(const std::string& variable_name) {
	vector<double> result = _data.at< std::vector<double> >(variable_name);
	
	simplifiedData sd = processMxArray(variable_name);
	if (isValidData(sd)) {
		mxArray *mxa = getVariable(variable_name);
		
		if (mxIsDouble(mxa)) {
			result = castVector<double, double>(sd);
		} else if ( mxIsSingle(mxa) ) {
			result = castVector<float, double>(sd);
		} else if (mxIsInt32(mxa)) {
			result = castVector<int, double>(sd);
		} else if (mxIsInt64(mxa)) {
			result = castVector<long long, double>(sd);
		} else if (mxIsUint32(mxa)) {
			result = castVector<unsigned int, double>(sd);
		} else if (mxIsUint64(mxa)) {
			result = castVector<unsigned long long, double>(sd);
		}

		
	}
	
	//std::cout << "getDoubleVector " << variable_name << " : " << viewVector(result) << std::endl;
	return result;
}
*/


/**
vector<float> MEXHandler::getFloatVector(const std::string& variable_name) {
	vector<float> result = _data.at< std::vector<float> >(variable_name);
	simplifiedData sd = processMxArray(variable_name);
	if (isValidData(sd)) {
		mxArray *mxa = getVariable(variable_name);

		if (mxIsDouble(mxa)) {
			result = castVector<double, float>(sd);
		} else if (mxIsSingle(mxa)) {
			result = castVector<float, float>(sd);
		} else if (mxIsInt32(mxa)) {
			result = castVector<int, float>(sd);
		} else if (mxIsInt64(mxa)) {
			result = castVector<long long, float>(sd);
		} else if (mxIsUint32(mxa)) {
			result = castVector<unsigned int, float>(sd);
		} else if (mxIsUint64(mxa)) {
			result = castVector<unsigned long long, float>(sd);
		}


	}

	//std::cout << "getFloatVector " << variable_name << " : " << viewVector(result) << std::endl;
	return result;
}	   
	*/


void MEXHandler::Flush_Major(const std::string& major) {
	auto it = _output_buffers.find(major);
	if (it != _output_buffers.end()) {
		std::ostringstream oss("");
		mxArray *mxa = it->second; // the output structure to be read
		MATFile *pmat;
		pmat = matOpen(major.c_str(), "w");
		if (pmat == NULL) {
			oss << "file, " << major << " can't be open, aborts";
			throw std::runtime_error(oss.str());
		}
		auto status = matPutVariable(pmat, "sout", mxa);
		if (status != 0) {
			oss << __FILE__ << " :  Error using matPutVariable on line " << __LINE__ << ", status: "<<status;
			throw std::runtime_error(oss.str());
		}
		if (matClose(pmat) != 0) {
			oss << "error, closing the file "<<major;
			throw std::runtime_error(oss.str());
		}
	}
}
 
/**
vector<int> MEXHandler::getIntVector(const std::string& variable_name) {
	vector<int> result = _data.at< std::vector<int> >(variable_name);
	simplifiedData sd = processMxArray(variable_name);
	if (isValidData(sd)) {
		mxArray *mxa = getVariable(variable_name);

		if (mxIsDouble(mxa)) {
			result = castVector<double, int>(sd);
		} else if (mxIsSingle(mxa)) {
			result = castVector<float, int>(sd);
		} else if (mxIsInt32(mxa)) {
			result = castVector<int, int>(sd);
		} else if (mxIsInt64(mxa)) {
			result = castVector<long long, int>(sd);
		} else if (mxIsUint32(mxa)) {
			result = castVector<unsigned int, int>(sd);
		} else if (mxIsUint64(mxa)) {
			result = castVector<unsigned long long, int>(sd);
		}


	}

	///std::cout << "getIntVector " << variable_name << " : " << viewVector(result) << std::endl;
	return result;
} 
*/
