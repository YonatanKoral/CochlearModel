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

#ifndef __BIN_FILE
#define __BIN_FILE

#include "mutual.h"
#include "cvector.h"
// Binary file open options (writing, reading and both):
enum bin_types { 
	BIN_READ		= std::ios::in | std::ios::binary, 
	BIN_WRITE		= std::ios::out | ios::trunc | std::ios::binary,
	BIN_READ_WRITE	= std::ios::out | std::ios::in | std::ios::binary
};


// Binary File is derived from ofstream 
class CBin
{

	bin_types		_bin_type;		// binary type, e.g. reading, writing etc.
	char		_filename[MAX_BUF_LENGTH];		// file name.
	std::fstream	_file;			// file handle.

public:

	double			_file_length;	// File's length

	// open bin file for reading/writing: 
	CBin(const std::string& filename, bin_types bin_type);
	
	// d'tor
	~CBin();

	// Insert\Append new message line
	CBin& operator <<(const double& scalar);
	CBin& operator <<(const vector<double>& v);

	// Read data from file (file must be open): 
	bool read_padd(const vector<double>& v, const long file_loc, long file_length);
	bool read(double& scalar);
	bool read(vector<double>& v);
	
	// Read data (one element) from file at a specific location (file must be open).
	//	Note: The file_loc is taken from the beginning.
	bool read(double& scalar, const long file_loc);

	//bool write(const double scalar);
	//bool write(const CVector scalar);

	void rewind(void);			// rewomd file to the beginning.

	bool close_file(void);		// close file.

	// if EOF, return true
	bool is_eof(void);

};

#endif


