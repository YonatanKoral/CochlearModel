/******************************************************************\

File Name : bin.h
===========

Classes Defined : CBin.
=================

Description	: defines a class to access binary files.
=============

Example :
=========
CBin _log_dat("dat.bin");		// Creating binary file.
_log_dat << SECTIONS;	// write <double> to binary log.
_log_dat << LowerDiag;	// write <CVector> to binary log.


\******************************************************************/
#pragma once

#ifndef __T_BIN_FILE
#define __T_BIN_FILE

#include "mutual.h"
#include "cvector.h"
#include "bin.h"
#include <iosfwd>
#include <ostream>
// Binary file open options (writing, reading and both):
#include <iostream>
#include <sstream>
#include <errno.h>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
inline void __errorMessagenger(const errno_t errcode , const char *file, const int line) {

	//if (0 != errcode) {
		
		cout << file << "(" << line << ") : error found - " << errcode<<"\n";
	//}
}
#define BOOST_MY_THROW(code) __errorMessagenger(code,__FILE__,__LINE__)

template<class T> class Print {
private:
	std::shared_ptr<std::fstream> _ofs;
	const int _max_length; // max length per print
public:
	Print(std::shared_ptr<std::fstream>& os, const int max_length) :_ofs(os), _max_length(max_length) {}
	Print(const Print& src) : _ofs(src._ofs), _max_length(src._max_length) {}
	Print& operator <<(const vector<T>& v);
};
template<class T> Print<T>& Print<T>::operator<<(const vector<T>& v) {
	_ofs->write((const char*)(&v[0]), __tmin(_max_length, int(v.size()))*sizeof(T));
	return (*this);
}
// Binary File is derived from ofstream 
template<class T> class TBin
{

	bin_types		_bin_type;		// binary type, e.g. reading, writing etc.
	char		_filename[MAX_BUF_LENGTH];		// file name.
	std::shared_ptr<std::fstream>	_file;			// file handle.
public:

	double			_file_length;	// File's length
	
	TBin(const std::string& filename, bin_types bin_type, const bool& to_open);

	// open bin file for reading/writing: 
	TBin(const std::string& filename, bin_types bin_type) : TBin(filename, bin_type, true) {}
	TBin(const TBin& src) : _file(src._file), _bin_type(src._bin_type) { strcpy_s<MAX_BUF_LENGTH>(_filename, src._filename);  }
	// d'tor
	~TBin();

	inline bool _is_opened() { return _file->is_open(); } // file can be opened on construction or not
	bool open_file();
	// Insert\Append new message line
	TBin& operator <<(const T& scalar);
	TBin& operator <<(const vector<T>& v);
	void limitWrite(const long& max_length, const vector<T>& v);
	// Read data from file (file must be open): 
	bool read_padd(vector<T>& v, const long file_loc, long file_length);
	bool read_padd(vector<T>& v, const long output_location, const long file_loc, long length_array);
	bool read(T& scalar);
	bool read(vector<T>& v);

	// Read data (one element) from file at a specific location (file must be open).
	//	Note: The file_loc is taken from the beginning.
	bool read(T& scalar, const long file_loc);

	//bool write(const double scalar);
	//bool write(const CVector scalar);

	void rewind(void);			// rewomd file to the beginning.

	bool close_file(void);		// close file.

	// if EOF, return true
	bool is_eof(void);

};
#include "TBin.tpp"
#endif


