/******************************************************************\

	File Name : error.h
	===========

	Classes Implemented:	MyError.

	Description	: a family of exceptions that hold error details
	=============
					the exception is thrown when an error occurs


\******************************************************************/


#include "error.h"

// c'tor
MyError::MyError() {}

// c'tor that gets a message
MyError::MyError(const std::string msg, const std::string title) : _title(title), _message(msg) {}

// copy c'tor (e.g. for the try-catch)
MyError::MyError(const MyError& other) : _message(other._message), _title(other._title) {}

// Insert\Append message
MyError& MyError::operator <<(std::string err_msg)
{
	_message.append(err_msg);
	return *this;
}


// Return full error message (title & message)  
std::string MyError::what(void)
{
	std::string full_error_line("[" + _title + "] " + _message + "\n");
	return full_error_line;
}


// d'tor
MyError::~MyError() {};

