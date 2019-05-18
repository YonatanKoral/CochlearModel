#pragma once
#include <sstream>
#include <iostream>
#include "mex.h"

class mxstreambuf : public std::streambuf {
public:
	mxstreambuf() {
		stdoutbuf = std::cout.rdbuf(this);
	}
	~mxstreambuf() {
		std::cout.rdbuf(stdoutbuf);
	}
protected:
	virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
		mexPrintf("%.*s", n, s);
		return n;
	}
	virtual int overflow(int c = EOF) override {
		if (c != EOF) {
			mexPrintf("%.1s", &c);
		}
		return 1;
	}
private:
	std::streambuf *stdoutbuf;
};
/*
typedef int(__cdecl * PrintFunc)(const char *, ...);

int __cdecl nullprintf(const char *, ...)
{
	//nothing is printed
	return 0;
}

template<int N, PrintFunc PRINTFUNC, int VERBOSE = 0>
class mxstreambuf : public streambuf
{
protected:

	char m_buffer[N];

public:

	// the constructor sets up the entire reserve
	// buffer to buffer output,... since there is
	// no input! ;-)

	mxstreambuf() : streambuf(m_buffer, N)
	{
		setp(m_buffer, m_buffer + N);
	}

	// outputs characters to the device via 
	// PRINTFUNC. since there is no input,
	// there isnt really anything to sync!

	int sync()
	{
		int n = out_waiting();

		if (!n) {
			return 0;
		}

		if (VERBOSE) {
			PRINTFUNC("n=%d\n", n);
		}

		xsputn(pbase(), n);

		pbump(-n);

		return 0;
	}

	// called when the associated buffer is 
	// full:

	int overflow(int ch)
	{
		sync();

		if (VERBOSE) {
			PRINTFUNC("OF:%c", ch);
		}
		else {
			PRINTFUNC("%c", ch);
		}

		return 0;
	}

	// VisualC requires that this be defined.
	// since there is no input available, return
	// EOF.

	int underflow()
	{
		return EOF;
	}

	// prints a series of characters to the
	// screen:

	int xsputn(char *text, int n)
	{
		if (!n) {
			return 0;
		}

		char printf_fmt[16];

		sprintf(printf_fmt, "%%.%ds", n);

		if (VERBOSE) {
			PRINTFUNC("format = %s\n", printf_fmt);
		}

		PRINTFUNC(printf_fmt, text);

		return 0;
	}
};
*/
