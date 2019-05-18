/******************************************************************\

	File Name : bin.h
	===========

	Classes Defined : CBin.
	=================

	Description	: defines a class to access binary files.
	=============

	Example:


	NOTES :
	=======
	The binary file can be access from MATLAB by the following commands:

		MATLAB:
			fid = fopen('log.dat', 'rb');
			y = fread( fid, Inf, 'double' );	% y holds a vector.

\******************************************************************/

#include "bin.h"

// opens a BINARY log file
CBin::CBin(const string& filename, bin_types bin_type)			
{

	strcpy_s<MAX_BUF_LENGTH>(_filename, filename.data());		// save the filename into the CBin class.
	_bin_type = bin_type;		// save the type of the binary file.

	// Open the CBin file for writing
	_file.open( _filename, _bin_type );

	//printf("Opening file %s\n",_filename.c_str()); getchar();

	if (!_file.is_open())
	{
		cout << "can't open binary log file "<< filename.data() << " - " << _filename <<"\n"; getchar(); 
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("can't open binary log file ", "CBin");
		//io_err << filename.c_str();
		//throw io_err;
	}

	// Get file's length
	_file.seekg(0, ios::end);
	_file_length = static_cast<double>(_file.tellg()) / sizeof(double);

	// Rewind to the beginning of the file
	rewind();

}

CBin::~CBin()
{
	if (_file.is_open())
		_file.close();
}

// Write/Append new message line
CBin& CBin::operator <<(const double& scalar)
{

	if ( (_bin_type != BIN_WRITE) && (_bin_type != BIN_READ_WRITE) )	{
		cout<<"CBin::operator <<(const double scalar) - Invalid <_bin_type>\n";  
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("CBin::operator <<(const double scalar) - Invalid <_bin_type>", "CBin");
		//throw io_err;
	}		

	_file.write( (const char*)(&scalar), sizeof(double) );
	_file.flush();
	return *this;

}

// Inserts a vector to the log	
CBin& CBin::operator <<(const vector<double>& v)
{
	if ( (_bin_type != BIN_WRITE) && (_bin_type != BIN_READ_WRITE) )	{
		cout<<"CBin::operator <<(const CVector v) - Invalid <_bin_type>\n";  
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("CBin::operator <<(const CVector v) - Invalid <_bin_type>", "CBin");
		//throw io_err;
	}		

	if (!_file.is_open())
	{
		cout<<"can't open file for writing\n";
		throw std::runtime_error("cant open save float array file");
		//MyError io_err("CBin::write(&CVector v) - Can't open binary file for writing!", "CBin");
		//throw io_err;
	}

	for (int i = 0;i < (int)v.size(); i++)
		_file.write( (const char*)(&v[i]), sizeof(double) );

	_file.flush();
	return *this;
}

// d'tor



bool CBin::read_padd(const vector<double>& v, const long file_loc, long length_array)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "CBin::read Binary file of this type isn't open for reading"<<endl;
		throw std::runtime_error("CBin::read Binary file of this type isn't open for reading - 3");
		//MyError io_err("CBin::read(double& scalar, const long file_loc) - Binary file of this type isn't open for reading", "CBin");
		//throw io_err;
	}		

	if (!_file.is_open())
	{
		cout << "CBin::read Binary file isn't open for reading"<<endl;
		throw std::runtime_error("CBin::read Binary file isn't open for reading - 3");
		//MyError io_err("CBin::read(double& scalar, const long file_loc) - Binary file isn't open for reading", "CBin");
		//throw io_err;
	}
	if ( _file_length <= file_loc ) 
	{
		
	return 1; 
	}

	_file.seekg( (sizeof(double)) * file_loc, ios::beg );

	if ( is_eof() )
	{

	return 1;
	}
	for ( int i = 0 ; i< min(length_array,(long)v.size());i++) {
			//*((char *)&scalar) = 0.0;
		 _file.read((char *)&(v[i]), sizeof(double));
		 if (sizeof(double) != _file.gcount())
		 {
			 cout << "read padd ended at " << i << "\n";
			 break;
			 //MyError io_err("CBin::read(&CVector v) - Can't read <vector> from the binary file", "CBin");
			 //throw io_err;
		 }
	}
	/*
	if ( sizeof scalar != _file.gcount() )
	{
		
		cout << "CBin::read_padd(&CVector v) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("CBin::read_padd(&CVector v) - Can't read <scalar> from the binary file - 4");
		//MyError io_err("CBin::read_padd(&CVector v) - Can't read <scalar> from the binary file", "CBin");
		//throw io_err;
	}
	*/
	return ( _file.good() );

}


bool CBin::read(double& scalar)
{
	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "CBin::read(double& scalar) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("CBin::read(double& scalar) - Invalid <_bin_type> - 5");
		//MyError io_err("CBin::read(double& scalar) - Invalid <_bin_type>", "CBin");
		//throw io_err;
	}		

	// DEBUG
	long aaa = static_cast<long>(_file.tellg()) / sizeof(double);
	bool bbb = _file.eof();
	_file_length;

	if (!_file.is_open())
	{
		cout << "CBin::read(&CVector v) - Can't open binary file for writing!"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Can't open binary file for writing! - 6");
		//MyError io_err("CBin::read(&CVector v) - Can't open binary file for writing!", "CBin");
		//throw io_err;
	}else if ( is_eof() )
	{
		cout << "CBin::read(&CVector v) - Binary file can't continue reading because EOF !"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Binary file can't continue reading because EOF ! - 7");
		//MyError io_err("CBin::read(&CVector v) - Binary file can't continue reading because EOF !", "CBin");
		//throw io_err;

	}

	_file.read( (char *)&scalar, sizeof scalar );
	if ( sizeof scalar != _file.gcount() )
	{
		cout << "CBin::read(&CVector v) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Can't read <scalar> from the binary file - 8");
		//MyError io_err("CBin::read(&CVector v) - Can't read <scalar> from the binary file", "CBin");
		//throw io_err;
	}

	return ( _file.good() );

}

bool CBin::read(vector<double>& v)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		cout << "CBin::read(&CVector v) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Invalid <_bin_type> - 9");
		//MyError io_err("CBin::read(CVector& v) - Invalid <_bin_type>", "CBin");
		//throw io_err;
	}		

	if (!_file.is_open())
	{
		cout << "CBin::read(&CVector v) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Binary file isn't open for reading - 10");
		//MyError io_err("CBin::read(&CVector v) - Binary file isn't open for reading", "CBin");
		//throw io_err;
	}else if ( is_eof() )		// ToDo - Also needs to check if the vector's length is longer then the actual available file size.
	{
		cout << "CBin::read(&CVector v) - Binary file can't continue reading because EOF"<<endl;
		throw std::runtime_error("CBin::read(&CVector v) - Binary file can't continue reading because EOF - 11");
		//MyError io_err("CBin::read(&CVector v) - Binary file can't continue reading because EOF !", "CBin");
		//throw io_err;

	}
	
	for (int i = 0; i < (int)v.size(); i++ )
	{
		_file.read( (char *)&(v[i]), sizeof(double) );

		if ( sizeof(double) != _file.gcount() )
		{
			cout << "CBin::read(&CVector v) - Can't read <vector> from the binary file"<<endl;
			throw std::runtime_error("CBin::read(&CVector v) - Can't read <vector> from the binary file - 12");
			//MyError io_err("CBin::read(&CVector v) - Can't read <vector> from the binary file", "CBin");
			//throw io_err;
		}
	}

	return ( _file.good() );

}

// Read data (one element) from file at a specific location (file must be open).
//	Note: The file_loc is taken from the beginning.
bool CBin::read(double& scalar, const long file_loc)
{

	if ( (_bin_type != BIN_READ) && (_bin_type != BIN_READ_WRITE) )	{
		
		cout << "CBin::read(double& scalar, const long file_loc) - Invalid <_bin_type>"<<endl;
		throw std::runtime_error("CBin::read(double& scalar, const long file_loc) - Invalid <_bin_type> - 13");
		//MyError io_err("CBin::read(double& scalar, const long file_loc) - Invalid <_bin_type>", "CBin");
		//throw io_err;
	}		

	if (!_file.is_open())
	{
		cout << "CBin::read(double& scalar, const long file_loc) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("CBin::read(double& scalar, const long file_loc) - Binary file isn't open for reading - 14");
		//MyError io_err("CBin::read(double& scalar, const long file_loc) - Binary file isn't open for reading", "CBin");
		//throw io_err;
	}

	if ( _file_length < file_loc ) 
	{
		cout << "CBin::read(double& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!! "<<endl;
		throw std::runtime_error("CBin::read(double& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!!  - 15");
		//MyError io_err("CBin::read(double& scalar, const long file_loc) - file position to read is out of bound (_file_length > file_loc) !!! ", "CBin");
		//throw io_err;
	}

	_file.seekg( (sizeof scalar) * file_loc, ios::beg );

	if ( is_eof() )
	{
		cout << "CBin::read(double& scalar, const long file_loc) - Binary file can't continue reading because EOF ! "<<endl;
		throw std::runtime_error("CBin::read(double& scalar, const long file_loc) - Binary file can't continue reading because EOF !  - 16");
		//MyError io_err("CBin::read(&CVector v) - Binary file can't continue reading because EOF !", "CBin");
		//throw io_err;
	}

	_file.read( (char *)&scalar , sizeof scalar );

	if ( sizeof scalar != _file.gcount() )
	{
		cout << "CBin::read(double& scalar, const long file_loc) - Can't read <scalar> from the binary file"<<endl;
		throw std::runtime_error("CBin::read(double& scalar, const long file_loc) - Can't read <scalar> from the binary file - 17");
		//MyError io_err("CBin::read(&CVector v) - Can't read <scalar> from the binary file", "CBin");
		//throw io_err;
	}

	return ( _file.good() );

}


void CBin::rewind(void)
{
	if (!_file.is_open())
	{
		cout << "CBin::rewind(void) - Binary file isn't open for reading"<<endl;
		throw std::runtime_error("CBin::rewind(void) - Binary file isn't open for reading - 18");
		//MyError io_err("CBin::rewind(void) - Binary file isn't open for reading", "CBin");
		//throw io_err;
	}
	_file.seekg(0, ios::beg);

}

bool CBin::close_file(void)
{
	if (_file.is_open())
		_file.close();

	return _file.good();
}

// if EOF, return true
bool CBin::is_eof(void)
{
	if ( _file_length == _file.tellg()/sizeof(double) )
		return true;
	else
		return false;

}
