

#include "CVector.h"
using namespace std;


template<class T> void truncZeros(vector<T, std::allocator<T>>& v){
	while (!v.empty() && v.back() == 0) {
		v.pop_back();
	}
}

template<class T> auto operator +(const vector<T, std::allocator<T>>& v_left, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{

	if ( v_left.size() != v_right.size() )
	{
		//MyError cv_err("The two vectors should be of same size ( v_left.size() != v_right.size() )", "operator +(vector<T, std::allocator<T>>, vector<T, std::allocator<T>>)");
		cout << "The two vectors should be of same size ( v_left.size() != v_right.size() ), operator +(vector<T, std::allocator<T>>, vector<T, std::allocator<T>>)" << endl;
		exit(1);
	}

	vector<T, std::allocator<T>> result(v_left.size(), 0);
	
	for (int i = 0; i < (int)v_left.size(); i++)
		result[i] = v_left[i] + v_right[i];

	return result;

}
template<class T> auto operator +(const vector<T, std::allocator<T>>& v_left, const T& scalar)	-> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_left.size());

	for (int i = 0; i < (int)v_left.size(); i++)
		result[i] = v_left[i] + scalar;

	return result;

}

template<class T> auto operator +(const T& scalar, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{
	return (v_right + scalar);
}

template<class T> auto operator -(const vector<T, std::allocator<T>>& v_left, const vector<T, std::allocator<T>>& v_right)	-> vector<T, std::allocator<T>>
{

	if ( v_left.size() != v_right.size() )
	{
		cout << "The two vectors should be of same size ( v_left.size() != v_right.size() ), operator +(vector<T, std::allocator<T>>, vector<T, std::allocator<T>>)" << endl;
		exit(1);
	}

	vector<T, std::allocator<T>> result(v_left.size(), 0);
	
	for (int i = 0; i < (int)v_left.size(); i++)
		result.at(i) = v_left.at(i) - v_right.at(i);

	return result;

}


template<class T> auto operator -(const vector<T, std::allocator<T>>& v_left, const T& scalar)	 -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_left.size());

	for (int i = 0; i < (int)v_left.size(); i++)
		result.at(i) = v_left.at(i) - scalar;

	return result;

}

template<class T> auto operator -(const T& scalar, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_right.size());

	for (int i = 0; i < (int)v_right.size(); i++)
		result.at(i) = scalar - v_right.at(i);

	return result;

}


// Multiplication:
template<class T> auto operator *(const vector<T, std::allocator<T>>& v_left, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{

	//assert( v_right.size() == v_left.size() );

	vector<T, std::allocator<T>> result(v_right.size());

	for (int i = 0; i < (int)v_right.size(); i++)
		result.at(i) = v_right.at(i) * v_left.at(i);

	return result;

}


// Convolution:
template<class T> auto conv(const vector<T, std::allocator<T>>& A, const vector<T, std::allocator<T>>& B)  -> vector<T, std::allocator<T>>
{

	int nconv;
	int i, j, i1;
	double tmp;

	//allocated convolution array	
	nconv = static_cast<int>(A.size() + B.size() - 1);
	vector<T, std::allocator<T>> C(nconv);

	//convolution process
	for (i = 0; static_cast<unsigned int>(i)<C.size(); i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; static_cast<unsigned int>(j)<B.size(); j++)
		{
			if (i1 >= 0 && static_cast<unsigned int>(i1)<A.size())
				tmp = tmp + (A.at(i1) * B.at(j));

			i1 = i1 - 1;
			C.at(i) = tmp;
		}
	}
	//return convolution array
	return(C);

}


template<class T> auto operator *(const vector<T, std::allocator<T>>& v_left, const T& scalar) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_left.size());

	for (int i = 0; i < (int)v_left.size(); i++)
		result.at(i) = v_left.at(i) * scalar;

	return result;
}

template<class T> auto operator *(const T& scalar, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{
	return (v_right * scalar);
}

// Division:
template<class T> auto operator /(const vector<T, std::allocator<T>>& v_left, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{

	if ( v_right.size() != v_left.size() )	{
		cout << "ERROR - vector<T, std::allocator<T>> - Both sides of the division should be of the same size !" << endl;
		exit(1);
	}

	vector<T, std::allocator<T>> result(v_right.size());

	for (int i = 0; i < (int)v_right.size(); i++)
		result.at(i) = v_left.at(i) / v_right.at(i);

	return result;

}


template<class T> auto operator /(const vector<T, std::allocator<T>>& v_left, const T& scalar) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_left.size());

	for (int i = 0; i < (int)v_left.size(); i++)
		result.at(i) = v_left.at(i) / scalar;

	return result;
}

template<class T> auto operator /(const T& scalar, const vector<T, std::allocator<T>>& v_right) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v_right.size());

	for (int i = 0; i < (int)v_right.size(); i++)
		result.at(i) = scalar / v_right.at(i);

	return result;
}




// Overloading the exp() function for the vector<T, std::allocator<T>> class
template<class T> auto exp(const vector<T, std::allocator<T>>& v) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v.size(), 0);

	for (int i = 0; i < (int)v.size(); i++ )
		result.at(i) = exp(v.at(i));

	return result;

}

template<class T> auto vmax(const vector<T, std::allocator<T>>& v) -> T
{
	T m = v.at(0);
	for (int i=1 ; i < (int)v.size(); i++)
		if (v.at(i) > m)
			m = v.at(i);

	return m;
}

template<class T> auto get(const vector<T, std::allocator<T>>& v, const int& i) -> T
{
    if( i > (int)v.size() )
    {
        cout << "Index out of bounds" <<endl; 
        // return first element.
		return v.at(0);
    }
	return v.at(i);
}
template<class T> auto vmax(const vector<T, std::allocator<T>>& v_left, const vector<T, std::allocator<T>>& v_righ) -> vector<T, std::allocator<T>>
{

	if ( v_left.size() != v_righ.size() )
	{
		cout << "ERROR - max(vector<T, std::allocator<T>>, vector<T, std::allocator<T>>) - the two vectors are of different length !" << endl;
		exit(1);
	}

	vector<T, std::allocator<T>> result(v_left.size(), 0);
	for (int i = 0; i < (int)v_left.size(); i++ )
		result[i] = __tmax( v_left[i], v_righ[i] );

	return result;

}
template<class T> auto vmax(const T& scalar, const vector<T, std::allocator<T>>& v) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v.size(), 0);
	for (int i = 0; i < (int)v.size(); i++ )
		result.at(i) = __tmax(v.at(i), scalar);

	return result;
}

template<class T> auto vmax(const vector<T, std::allocator<T>>& v, const T& scalar) -> vector<T, std::allocator<T>>
{
	return vmax( scalar, v );
}

template<class T> auto sum(const vector<T, std::allocator<T>>& v) -> T
{
	T scalar = v.size()>0?v[0]:NULL;
	for ( int i=1 ; i < (int)v.size(); scalar += v.at(i++) );
	return scalar;
}
template<class T,class T2> auto partialSum(const vector<T, std::allocator<T>>& v, const int& target_size)->vector<T2, std::allocator<T2>> {
	vector<T2, std::allocator<T2>> result(target_size, T2(0));
	int sector_size =static_cast<int>(v.size()) / target_size;
	int sector_index = 0;
	for (int i = 0; i < static_cast<int>(v.size()); i++) {
		if (i > 0 && i%sector_size == 0) sector_index++;
		result[sector_index] += static_cast<T2>(v.at(i));
	}
	return result;
}


template<class T, class T2> auto partialAvg(const vector<T, std::allocator<T>>& v, const int& target_size)->vector<T2, std::allocator<T2>> {
	vector<T2, std::allocator<T2>> result = partialSum<T, T2>(v, target_size);
	int sector_size = static_cast<int>(v.size()) / target_size;
	int last_sector = static_cast<int>(result.size()) - 1;
	for (int i = 0; i < last_sector; i++) {
		result[i] = result[i] / static_cast<T2>(sector_size);
	}
	int last_sector_size = static_cast<int>(v.size()) - last_sector*sector_size;
	result[last_sector] = result[last_sector] / static_cast<T2>(last_sector_size);
	return result;
}

template<class T> auto abs(const vector<T, std::allocator<T>>& v) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> result(v.size(), 0);
	for (int i = 0; i < (int)v.size(); i++ )
		result.at(i) = abs(v.at(i));

	return result;
}


//// count the number of positions in which elements in v1
//// are greater than their corresponding v2 elements
//int count_greater(const vector<T, std::allocator<T>> v1, const vector<T, std::allocator<T>> v2)
//{
//	int cnt = 0;
//	for ( int i=0; i < v2.size(); i++)
//		cnt += (v1[i] > v2[i]);
//	return cnt;
//}
template<class T> auto operator ^(const vector<T, std::allocator<T>>& v, const T& p) -> vector<T, std::allocator<T>>
{

	

	vector<T, std::allocator<T>> vv = v;
	for (int i = 0; i < (p-1); i++)
		vv = vv * v;

	return vv;

}

template<class T> auto sqrt(const vector<T, std::allocator<T>>& v) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> vv(v.size(), 0);

	for (int i = 0; i < (int)v.size(); i++)
		vv[i] = T(sqrt(static_cast<double>(v[i])));

	return vv;
}

template<class T> auto tanh(const vector<T, std::allocator<T>>& v) -> vector<T, std::allocator<T>>
{
	vector<T, std::allocator<T>> vv(v.size(), 0);

	for (int i = 0; i < (int)v.size(); i++)
		vv[i] = T(tanh(static_cast<double>(v[i])));

	return vv;
}
/**
template<class T> void self_add(std::vector<T, std::allocator<T>>& v, const size_t& start_index, const size_t& end_index, const T& value) {
	std::vector<T, std::allocator<T>>::iterator stv = std::next(v.begin(),start_index);
	std::vector<T, std::allocator<T>>::iterator etv = std::next(stv, end_index-start_index+1);
	std::transform(stv, etv, stv, [&value](T loop_val) { return (loop_val + value); });
}
*/
template<class T> string viewVector(const vector<T, std::allocator<T>>& v, const viewVectorModes& modes) {
	int base_size = (int)v.size();
	if (modes.line_seperation_every_x_values > 0) base_size += base_size / modes.line_seperation_every_x_values;
	ostringstream oss;
	oss.str("");
	oss.setf(oss.boolalpha);
	oss.setf(modes.float_flags);
	oss.precision(modes.precision);
	for (int i = 0; i < (int)v.size(); i++) {
		if (i != 0) oss << " ";
		oss << v.at(i);
		if (modes.line_seperation_every_x_values > 0 && (i + 1) % modes.line_seperation_every_x_values == 0) {
			oss << "\n";
		}
	}
	return oss.str();
}

template<class T> string viewVector(const vector<T, std::allocator<T>>& v, const int& line_seperation_every_x_values) {
	// viewVectorModes{ 0, std::ios_base::scientific, 1 }
	// (viewVectorModes) { .line_seperation_every_x_values = line_seperation_every_x_values, .float_flags = std::ios_base::scientific, .precision = 3 }
	return viewVector<T>(v, viewVectorModes{ line_seperation_every_x_values, std::ios_base::scientific, 3 });
}


template<class T> string viewVector(const vector<T, std::allocator<T>>& v) {
	return viewVector<T>(v, 0);
}

//template<class T> auto stepVector(const vector<T, std::allocator<T>>& v, const size_t& step, const size_t& first_index)->vector < T, std::allocator<T> {

//}


template<class T, class T2> string verboseVectors(const string& prefix,const int& precision,const int& seperation_frequency,const vector<T, std::allocator<T>>& v, const vector<T2, std::allocator<T2>>& v2) {
	int target_size = static_cast<int>(v2.size());
	int source_size = static_cast<int>(v.size());
	int sector_size = source_size / target_size;
	//cout << target_size << "," << source_size << "," << sector_size << endl;
	ostringstream oss("");
	oss.setf(oss.boolalpha);
	oss.setf(oss.scientific);
	oss.precision(precision);
	int target_index = 0;
	for (int i = 0; i < source_size; i++) {
		if (i%sector_size == 0) {
			if (i > 0) { 
				oss << "];" << endl;
				target_index++;
			}
			oss << prefix << "("<<target_index<<","<<v2[target_index]<<") = ["<< endl;

		}
		oss << v[i];
		if (i < source_size-1 && (i + 1) % sector_size != 0) oss << ", ";
		if ((i + 1) % seperation_frequency == 0) oss << endl;
	}
	return oss.str();
}
template <class T> auto parseToScalar(const string &Text) -> T {	 //Text not by const reference so that the function can be used with a 
	//character array as argument
	stringstream ss(Text);
	T result;
	return ss >> result ? result : 0;
}

template <typename T> auto lexicalCast(T Number)->string {
	stringstream ss;
	ss << Number;
	return ss.str();
}
template<class T> auto parseToVector(const string& str)->vector<T, std::allocator<T>> {
	std::stringstream ss(str);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;
	std::vector<std::string> vstrings(begin, end);
	vector<T, std::allocator<T>> result = vector<T, std::allocator<T>>(vstrings.size());
	
	std::transform(vstrings.begin(), vstrings.end(), result.begin(), parseToScalar<T>);
	return result;
}

template<class T1, class T2> auto castVector(const vector<T1, std::allocator<T1>>& v)->vector<T2, std::allocator<T2>> {
	vector<T2, std::allocator<T2>> result;
	std::transform(v.begin(), v.end(),
                    std::back_inserter(result),
                    CastToT2<T1,T2>());
	// = vector<T2, std::allocator<T2>>(v.begin(),v.end());
	return result;
}

template<class T1, class T2> auto castVector(const T1 *v, size_t vsize)->vector<T2, std::allocator<T2>> {
	vector<T2, std::allocator<T2>> result;
	std::transform(v, v+vsize,
		std::back_inserter(result),
		CastToT2<T1, T2>());
	// = vector<T2, std::allocator<T2>>(v.begin(),v.end());
	return result;
}

template<class T> bool checkVector(const vector<T, std::allocator<T>>& v, const T& value) {
	return !v.empty() && std::find(v.begin(), v.end(), value) != v.end();
}
template<class T1, class T2,class T3> void vectorSumTemplate(const vector<T1>& A,const T1& c1,const vector<T2>& B,const T2& c2,vector<T3>& C) {
	for (size_t sz=0;sz<A.size();sz++) {
		C[sz] = c1*A[sz] +  c2*B[sz];
		
	}
}

template<class T1, class T2, class T3> void FIRFilterTemplate(const vector<T1>& X, vector<T2>& Y, const vector<T3>& filter,int sections, int lambdaOffset, size_t time_dimension, int starts) {
	int time_block = static_cast<int>(time_dimension) / starts;
	int current_offset, offset_boundary, k, i;
	int max_offset = 0;
	int filter_size = static_cast<int>(filter.size());
	int time_length_analysis = time_block - lambdaOffset;

	//cout << "FIR filter CPU: blocks divided: " << starts << ", time_block length measured: " << time_block << ", time block length analysis: " << time_length_analysis;
	//printf("cpp time length analysis %d on starts=%d on sections=%d \n",time_length_analysis,starts,sections);
	for (int s = 0; s < sections; s++) {
		int cochlea_offset_section = s; // s*time_dimension; untransposed sections offset is not block, diffrent section every node
		for (int blockId = 0; blockId < starts; blockId++) {
			int time_block_start_offset = blockId*time_block;
			int offset = cochlea_offset_section + sections*(time_block_start_offset + lambdaOffset); // untransposed sections added for each block lambda offset unchanged
			for (k = 0; k < time_length_analysis; k++) {
				current_offset = offset + k*sections; // untransposed adding sections multiplication for k, offset time
				offset_boundary = std::min(time_block_start_offset + k + 1, filter_size);
				Y[current_offset] = 0;
				max_offset = max(current_offset, max_offset);
				for (i = 0; i < offset_boundary; i++) {
					Y[current_offset] = Y[current_offset] + static_cast<T2>(static_cast<T1>(filter[i]) * X[current_offset - i*sections]); // untransposed jumping by sections each time
				}
			}
		}
	}
}
template<class T1, class T2, class T3> void IIRFilterTemplate(const vector<T1>& X, vector<T2>& Y, const vector<T3>& a, const vector<T3>& b, int sections, int lambdaOffset, size_t time_dimension, int starts) {
	int time_block = static_cast<int>(time_dimension)/starts;
	int filter_size = static_cast<int>(a.size());
	int time_length_analysis = time_block - lambdaOffset;
	int max_offset = 0;
	/*
	std::cout << "time_length_analysis: " << time_length_analysis << ",sections: " << sections 
		<< ",starts: " << starts 
		<< ",lambda offset: " << lambdaOffset 
		<< ",time_block: " << time_block 
		<< ",filter_size: " << filter_size
		<< ",a = " << viewVector<T3>(a)
		<< ",b = " << viewVector<T3>(b)
		<< endl;
		*/
	int current_offset,offset_boundary,offset_boundarya,k,i,j;
	for(int s=0;s<sections;s++) {
		int cochlea_offset_section = s; //s*time_dimension; untransposed its just the offset from the beggining
		for(int blockId=0;blockId<starts;blockId++) {
			int time_block_start_offset = blockId*time_block;
			int offset = cochlea_offset_section + sections*(time_block_start_offset+lambdaOffset); // untransposed sections added for each block lambda offset unchanged
			for(k=0;k<time_length_analysis;k++){
				current_offset = offset+k*sections; // untransposed adding sections multiplication for k, offset time
				offset_boundary = min(time_block_start_offset+k+1,filter_size);
				offset_boundarya = min(k, filter_size - 1);
				max_offset = __tmax(current_offset, max_offset);
				Y[current_offset] = 0;
				for(i=0;i<offset_boundary;i++) {
					Y[current_offset] = Y[current_offset] + static_cast<T2>(static_cast<T1>(b[i]) * X[current_offset - i*sections]); // untransposed jumping by sections each time
				}
				for(i=0;i<offset_boundarya;i++) {
					j=i+1;
					Y[current_offset] = Y[current_offset] - static_cast<T2>(a[j])*Y[current_offset-j*sections]; // untransposed jumping by sections each time
				}
				
			}
		}
	}
	//cout << "max offset found " << max_offset << "\n";
}
template<class T> auto transposeVector(const std::vector<T, std::allocator<T>>& v,const size_t& column_size)->std::vector<T, std::allocator<T>> {
	std::vector<T> result(v.size());
	const size_t row_size = v.size() / column_size;
	//std::cout << "transposing  size  = " << v.size() << "(row_size=" << row_size << ",column_size=" << column_size<<")" << std::endl;
	for (size_t i = 0; i < v.size(); i++) {
		size_t ccol = i%row_size;
		size_t crow = i / row_size;
		result[ccol*column_size + crow] = v[i];
	}
	return result;
}

template<class T> auto transposeVector(const std::vector<T, std::allocator<T>>& v, const size_t& column_size, const size_t& partitions)->std::vector<T, std::allocator<T>> {
	std::vector<T> result(v.size());
	const size_t row_size = v.size() / (column_size*partitions);
	//std::cout << "transposing  size  = " << v.size() << "(row_size=" << row_size << ",column_size=" << column_size<<")" << std::endl;
	size_t partition_size = (v.size() / partitions);
	for (size_t partition_id = 0; partition_id < partitions; partition_id++) {
		size_t partition_offset = partition_id * partition_size;
		for (size_t i = 0; i < partition_size; i++) {
			size_t ccol = i%row_size;
			size_t crow = i / row_size;
			result[ccol*column_size + crow + partition_offset] = v[i + partition_offset];
		}
	}
	return result;
}

template<class keyClass, class valueClass> std::vector<keyClass> getAllKeys(const std::map<keyClass, valueClass>& m) {
	std::vector<keyClass> v;
	for (const auto &elem : m) {
		v.push_back(elem.first);
	}
	return v;
}


template<class keyClass, class valueClass> std::vector<keyClass> getAllValues(const std::map<keyClass, valueClass>& m) {
	std::vector<valueClass> v;
	for (const auto &elem : m) {
		v.push_back(elem.second);
	}
	return v;
}
template<typename T, typename... Args> void addToStream(std::ostringstream& a_stream, T&& a_value, Args&&... a_args) {
	a_stream << std::forward<T>(a_value);
	addToStream(a_stream, std::forward<Args>(a_args)...);
}

template< typename... Args > std::string concat(Args&&... a_args) {
	std::ostringstream s;
	addToStream(s, std::forward< Args >(a_args)...);
	return s.str();
}

template< typename T1, typename T2, typename PredicateT > void erase_if(std::map<T1, T2>& items, const PredicateT& predicate) {
	/*
	for (auto it = items.begin(); it != items.end();) {
		if (predicate(*it)) it = items.erase(it);
		else ++it;
	}
	*/
	auto it = std::find_if(items.begin(), items.end(), predicate);
	while ( it != items.end()) {
		items.erase(it);
		it = std::find_if(items.begin(), items.end(), predicate);
	}
}
template<class T> auto replicateVector(const std::vector<T, std::allocator<T>>& v, const size_t& times)->std::vector<T, std::allocator<T>> {
	std::vector<T, std::allocator<T>> result;
	for (size_t i = 0; i < times; i++) {
		for (auto vx : v) {
			result.push_back(vx);
		}
	}
	return result;
}

template<class T> auto expandVector(const std::vector<T, std::allocator<T>>& v, const size_t& times)->std::vector<T, std::allocator<T>> {
	std::vector<T, std::allocator<T>> result;
	
	for (auto vx : v) {
		for (size_t i = 0; i < times; i++) {
			result.push_back(vx);
		}
	}
	return result;
}

template<class T> auto expandVectorToSize(const std::vector<T, std::allocator<T>>& v, const size_t& to_size)->std::vector<T, std::allocator<T>> {
	std::vector<T, std::allocator<T>> result;
	size_t times = to_size/v.size();
	size_t remainder = to_size%v.size();
	size_t cindex = 0;
	for (auto vx : v) {
		for (size_t i = 0; i < times; i++) {
			result.push_back(vx);
		}
		if (cindex < remainder) {
			result.push_back(vx);
		}
		cindex++;
	}
	return result;
}
template<class T> auto replicateAndxpandVector(const std::vector<T, std::allocator<T>>& v, const size_t& times_replicate, const size_t& times_expand)->std::vector<T, std::allocator<T>> {
	std::vector<T, std::allocator<T>> result(v);
	if (result.size() > 0 && times_replicate / result.size() > 1) {
		result = replicateVector(result, times_replicate / result.size());
	}
	if (result.size() > 0 && times_expand / result.size() > 1) {
		result = expandVector(result, times_expand / result.size());
	}
	return result;
}