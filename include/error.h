/******************************************************************\

	File Name : error.h
	===========

	Classes Defined :	MyError.

	Description	: a family of exceptions that hold error details
	=============
					the exception is thrown when an error occurs
	Example:

		MyError io_err("It's an ERROR", "TITLE 1");
		cout << io_err.what();


\******************************************************************/
#pragma once

#ifndef __MY_ERROR
#define __MY_ERROR

#include "mutual.h"


// MyError is the basic error exception class
// it holds a message string that will be printed to check
// what happened.
// The other classes only put some default text inside this 
// message string. The user can then add his own.
class MyError
{
	std::string _title;			// The error's title.
	std::string _message;		// The error's message.

public:

	// c'tor
	MyError();

	// c'tor that get a string 
	//MyError(const char* msg = "", const char* title = "");
	MyError(const std::string msg, const std::string title);
	
	// copy c'tor
	MyError(const MyError& other);

	// Insert\Append message
	MyError& operator <<(std::string err_msg);

	// Return full error message (title & message)  
	std::string what(void);

	// d'tor
	~MyError();

};

#endif