/******************************************************************\

	File Name : bin.h
	===========

	Classes Defined : TBin.
	=================

	Description	: defines a class to access binary files.
	=============

	Example:


	NOTES :
	=======
	The binary file can be access from MATLAB by the following commands:

		MATLAB:
			fid = fopen('log.dat', 'rb');
			y = fread( fid, Inf, 'T' );	% y holds a vector.

\******************************************************************/

#include "TBin.h"

// opens a BINARY log file
template<class T> TBin<T>::TBin(const string& filename, bin_types bin_type, const bool& to_open)
{
	strcpy_s<MAX_BUF_LENGTH>(_filename, filename.data());		// save the filename into the TBin class.
	_bin_type = bin_type;		// save the type of the binary file.
	_file = std::make_shared<std::fstream>();
	if ( to_open) {
		open_file();
	}

}


template<class T> TBin<T>::~TBin()
{
	if (_file->is_open())
		_file->close();
}



template<class T> bool TBin<T>::open_file() {
	if (!_is_opened()) {
		// Open the TBin file for writing
		_file->open(_filename, _bin_type);

		//printf("Opening file %s\n",_filename.c_str()); getchar();

		if (!_file->is_open()) {
			cout << "can't open binary log file " << _filename << "\n"; 
			throw std::runtime_error("cant open save float array file");
			//MyError io_err("can't open binary log file ", "TBin");
			//io_err << filename.c_str();
			//throw io_err;
		}

		// Get file's length
		_file->seekg(0, ios::end);
		_file_length = static_cast<T>(_file->tellg()) / sizeof(T);

		// Rewind to the beginning of the file
		rewind();
	}
	return _is_opened();
}

// Write/Append new message line
template<class T> TBin<T>& TBin<T>::operator <<(const T& scalar)
{

	if ( (_bin_type != BIN_WRITE) && (_bin_type != BIN_READ_WRITE) )	{
		cout<<"TBin::operator <<(const T scalar) - Invalid <_bin_type>\n";  
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("TBin::operator <<(const T scalar) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}		

	_file->write( (const char*)(&scalar), sizeof(T) );
	_file->flush();
	return *this;

}

// Inserts a vector to the log	
template<class T> TBin<T>& TBin<T>::operator <<(const vector<T>& v)
{
	if ( (_bin_type != BIN_WRITE) && (_bin_type != BIN_READ_WRITE) )	{
		cout<<"TBin::operator <<(const CVector v) - Invalid <_bin_type>\n";  
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("TBin::operator <<(const CVector v) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}		

	if (!_file->is_open())
	{
		cout<<"can't open file for writing\n";
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("TBin::write(&CVector v) - Can't open binary file for writing!", "TBin");
		//throw io_err;
	}

	for (int i = 0;i < (int)v.size(); i++)
		_file->write( (const char*)(&v[i]), sizeof(T) );

	_file->flush();
	return *this;
}

// d'tor

template<class T> void TBin<T>::limitWrite(const long& max_length, const vector<T>& v) {
	if ((_bin_type != BIN_WRITE) && (_bin_type != BIN_READ_WRITE)) {
		cout << "TBin::operator <<(const CVector v) - Invalid <_bin_type>\n";
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("TBin::operator <<(const CVector v) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}

	if (!_file->is_open()) {
		cout << "can't open file for writing\n";
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("TBin::write(&CVector v) - Can't open binary file for writing!", "TBin");
		//throw io_err;
	}
	//for (int i = 0; i < (int)v.size(); i++)
	_file->write(reinterpret_cast<const char*>(v.data()), max_length*sizeof(T));
	//for (int i = 0; i < static_cast<int>(max_length); i++)
		//_file->write((const char*)(&v[i]), sizeof(T));
	_file->flush();
}

template<class T> bool TBin<T>::read_padd(vector<T>& v, const long output_location, const long file_loc, long length_array)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "TBin::read Binary file of this type isn't open for reading"<<endl;
		throw std::runtime_error("TBin::read Binary file of this type isn't open for reading - 3");
		//MyError io_err("TBin::read(T& scalar, const long file_loc) - Binary file of this type isn't open for reading", "TBin");
		//throw io_err;
	}		

	if (!_file->is_open())
	{
		cout << "TBin::read Binary file isn't open for reading"<<endl;
		throw std::runtime_error("TBin::read Binary file isn't open for reading - 3");
		//MyError io_err("TBin::read(T& scalar, const long file_loc) - Binary file isn't open for reading", "TBin");
		//throw io_err;
	}
	if ( _file_length <= file_loc ) 
	{
		
	return 1; 
	}

	_file->seekg( (sizeof(T)) * file_loc, ios::beg );

	if ( is_eof() )
	{

	return 1;
	}
	// read
	_file->read((char *)&v[output_location], __tmin(int(v.size()), int(__tmin(length_array,_file_length - file_loc)))*sizeof(T));
	
	/*
	for ( int i = 0 ; i< min(length_array,(long)v.size());i++) {
			//*((char *)&scalar) = 0.0;
		 
		 if (sizeof(T) != _file->gcount())
		 {
			 cout << "read padd ended at " << i << "\n";
			 break;
			 //MyError io_err("TBin::read(&CVector v) - Can't read <vector> from the binary file", "TBin");
			 //throw io_err;
		 }
	}
	*/
	/*
	if ( sizeof scalar != _file->gcount() )
	{
		
		cout << "TBin::read_padd(&CVector v) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("TBin::read_padd(&CVector v) - Can't read <scalar> from the binary file - 4");
		//MyError io_err("TBin::read_padd(&CVector v) - Can't read <scalar> from the binary file", "TBin");
		//throw io_err;
	}
	*/
	return ( _file->good() );

}
template<class T> bool TBin<T>::read_padd(vector<T>& v, const long file_loc, long length_array) {
	return read_padd(v, 0, file_loc, length_array);
}

template<class T> bool TBin<T>::read(T& scalar)
{
	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "TBin::read(T& scalar) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("TBin::read(T& scalar) - Invalid <_bin_type> - 5");
		//MyError io_err("TBin::read(T& scalar) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}		

	// DEBUG
	long aaa = static_cast<long>(_file->tellg()) / sizeof(T);
	bool bbb = _file->eof();
	_file_length;

	if (!_file->is_open())
	{
		cout << "TBin::read(&CVector v) - Can't open binary file for writing!"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Can't open binary file for writing! - 6");
		//MyError io_err("TBin::read(&CVector v) - Can't open binary file for writing!", "TBin");
		//throw io_err;
	}else if ( is_eof() )
	{
		cout << "TBin::read(&CVector v) - Binary file can't continue reading because EOF !"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Binary file can't continue reading because EOF ! - 7");
		//MyError io_err("TBin::read(&CVector v) - Binary file can't continue reading because EOF !", "TBin");
		//throw io_err;

	}

	_file->read( (char *)&scalar, sizeof scalar );
	if ( sizeof scalar != _file->gcount() )
	{
		cout << "TBin::read(&CVector v) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Can't read <scalar> from the binary file - 8");
		//MyError io_err("TBin::read(&CVector v) - Can't read <scalar> from the binary file", "TBin");
		//throw io_err;
	}

	return ( _file->good() );

}

template<class T> bool TBin<T>::read(vector<T>& v)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "TBin::read(&CVector v) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Invalid <_bin_type> - 9");
		//MyError io_err("TBin::read(CVector& v) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}		

	if (!_file->is_open())
	{
		cout << "TBin::read(&CVector v) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Binary file isn't open for reading - 10");
		//MyError io_err("TBin::read(&CVector v) - Binary file isn't open for reading", "TBin");
		//throw io_err;
	}else if ( is_eof() )		// ToDo - Also needs to check if the vector's length is longer then the actual available file size.
	{
		cout << "TBin::read(&CVector v) - Binary file can't continue reading because EOF"<<endl;
		throw std::runtime_error("TBin::read(&CVector v) - Binary file can't continue reading because EOF - 11");
		//MyError io_err("TBin::read(&CVector v) - Binary file can't continue reading because EOF !", "TBin");
		//throw io_err;

	}
	
	for (int i = 0; i < (int)v.size(); i++ )
	{
		_file->read( (char *)&(v[i]), sizeof(T) );

		if ( sizeof(T) != _file->gcount() )
		{
			cout << "TBin::read(&CVector v) - Can't read <vector> from the binary file"<<endl;
			throw std::runtime_error("TBin::read(&CVector v) - Can't read <vector> from the binary file - 12");
			//MyError io_err("TBin::read(&CVector v) - Can't read <vector> from the binary file", "TBin");
			//throw io_err;
		}
	}

	return ( _file->good() );

}

// Read data (one element) from file at a specific location (file must be open).
//	Note: The file_loc is taken from the beginning.
template<class T> bool TBin<T>::read(T& scalar, const long file_loc)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		
		cout << "TBin::read(T& scalar, const long file_loc) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("TBin::read(T& scalar, const long file_loc) - Invalid <_bin_type> - 13");
		//MyError io_err("TBin::read(T& scalar, const long file_loc) - Invalid <_bin_type>", "TBin");
		//throw io_err;
	}		

	if (!_file->is_open())
	{
		cout << "TBin::read(T& scalar, const long file_loc) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("TBin::read(T& scalar, const long file_loc) - Binary file isn't open for reading - 14");
		//MyError io_err("TBin::read(T& scalar, const long file_loc) - Binary file isn't open for reading", "TBin");
		//throw io_err;
	}

	if ( _file_length < file_loc ) 
	{
		cout << "TBin::read(T& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!! "<<endl;
		throw std::runtime_error("TBin::read(T& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!!  - 15");
		//MyError io_err("TBin::read(T& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!! ", "TBin");
		//throw io_err;
	}

	_file->seekg( (sizeof scalar) * file_loc, ios::beg );

	if ( is_eof() )
	{
		cout << "TBin::read(T& scalar, const long file_loc) - Binary file can't continue reading because EOF ! "<<endl;
		throw std::runtime_error("TBin::read(T& scalar, const long file_loc) - Binary file can't continue reading because EOF !  - 16");
		//MyError io_err("TBin::read(&CVector v) - Binary file can't continue reading because EOF !", "TBin");
		//throw io_err;
	}

	_file->read( (char *)&scalar , sizeof scalar );

	if ( sizeof scalar != _file->gcount() )
	{
		cout << "TBin::read(T& scalar, const long file_loc) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("TBin::read(T& scalar, const long file_loc) - Can't read <scalar> from the binary file - 17");
		//MyError io_err("TBin::read(&CVector v) - Can't read <scalar> from the binary file", "TBin");
		//throw io_err;
	}

	return ( _file->good() );

}


template<class T> void TBin<T>::rewind(void)
{
	if (!_file->is_open())
	{
		cout << "TBin::rewind(void) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("TBin::rewind(void) - Binary file isn't open for reading - 18");
		//MyError io_err("TBin::rewind(void) - Binary file isn't open for reading", "TBin");
		//throw io_err;
	}
	_file->seekg(0, ios::beg);

}

template<class T> bool TBin<T>::close_file(void)
{
	if (_file->is_open())
		_file->close();

	return _file->good();
}

// if EOF, return true
template<class T> bool TBin<T>::is_eof(void)
{
	if ( _file_length == _file->tellg()/sizeof(T) )
		return true;
	else
		return false;

}
