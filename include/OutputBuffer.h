#pragma once
#ifndef _OUTPUTBUFFER_H
#define _OUTPUTBUFFER_H
#include "TBin.h"
#include <string>
#include <iterator>
#include <algorithm>
template<class T> class OutputBuffer {
private:
	vector<T> _buffer;
	TBin<T> _file_handle;
	bool _append_mode;
	string _file_name;
	long _max_buffer_length;
	long _buffer_internal_position; // next position to write for the buffer
public:
	OutputBuffer(const std::string& fileName, const long& max_buffer_length, const bool& append_mode) :
		_file_name(fileName),
		_max_buffer_length(max_buffer_length),
		_buffer_internal_position(0L),
		_buffer(max_buffer_length),
		_append_mode(append_mode),
		_file_handle(fileName, append_mode ? BIN_READ_WRITE : BIN_WRITE, false) {
		PrintFormat("Buffer initialized: %s \n", fileName.c_str());
	}
	OutputBuffer() : OutputBuffer("", 0, false) {}
	OutputBuffer(const OutputBuffer& src) :
		_buffer(src._buffer),
		_append_mode(src._append_mode),
		_file_name(src._file_name),
		_max_buffer_length(src._max_buffer_length),
		_buffer_internal_position(src._buffer_internal_position),
		_file_handle(src._file_handle) {

	}
	~OutputBuffer() {
		PrintFormat("Buffer destroyed\n ");
	}
	inline bool boundaryClose(long length_from_boundary) { return _max_buffer_length <= _buffer_internal_position + length_from_boundary; }
	inline bool buffer_ready() { return _file_handle._is_opened(); }
	void init_buffer() {
		if (!buffer_ready()) {
			PrintFormat("Buffer opened: %s\n" ,_file_name.c_str() );
			_file_handle.open_file();
		}
	}
	void flush_buffer() {
		PrintFormat("Buffer flushed: %s,_buffer_internal_position = %d,_max_buffer_length = %d\n", _file_name.c_str(), _buffer_internal_position, _max_buffer_length);
		init_buffer();
		if (_buffer_internal_position > 0) {
			_file_handle.limitWrite(_buffer_internal_position, _buffer);
		}
		_buffer_internal_position = 0;
	}
	void defendBuffer(const int& length) {
		if (length > _max_buffer_length) {
			PrintFormat("cant contain current buffer extension in progress... %d/%d\n", _max_buffer_length, length);
			_max_buffer_length = 4 * length;
			_buffer.resize(_max_buffer_length);
		}
	}
	void append_buffer(T *src, const int length, const int offset) {
		defendBuffer(length);
		PrintFormat("buffer size allocated is %d reading %d positions, from internal postion: %d\n", _buffer.size(), length, _buffer_internal_position);
		if (_buffer_internal_position + length > _max_buffer_length) {
			flush_buffer();
		}
		//data_claimer(&_buffer[_buffer_internal_position], boost::numeric_cast<int>(length), boost::numeric_cast<int>(offset));

		//errno_t err = memcpy_s(&_buffer[_buffer_internal_position], (_buffer.size() - _buffer_internal_position)*sizeof(T), &src[offset], length*sizeof(T));
		auto bstart = std::next(_buffer.begin(), _buffer_internal_position);

		std::copy(&src[offset], &src[offset + length],bstart );
		/*
		BOOST_MY_THROW(err);
		cout << "from middle\n";
		for (int i = 0; i < 255; i++) {
		cout << _buffer[_buffer_internal_position + length / 2] << " ";
		if (i % 16 == 0 && i> 0) cout << "\n";
		}
		*/
		_buffer_internal_position += length;
		//std::cout << "copy " << length << " nodes to output buffer, buffer reached " << _buffer_internal_position << " internal position, for file name " << _file_name << "\n";
		//cout << "buffer size copied is " << length << " positions, internal position is now: " << _buffer_internal_position << "\n";
	}	// this overload from cuda read function
	OutputBuffer& operator=(const OutputBuffer& other) {
		// check for self-assignment
		if (&other == this)
			return *this;
		_buffer.resize(other._buffer.size());
		_append_mode = other._append_mode;
		_file_handle = other._file_handle;
		_file_name  = other._file_name;
		_max_buffer_length = other._max_buffer_length;
		std::copy(other._buffer.begin(), other._buffer.end(), _buffer.begin());
		return *this;
	}

	void append_buffer(const std::vector<T>& v, const int length, const int offset) {
		defendBuffer(length);
		PrintFormat("buffer size allocated is %d reading %d positions.. vector.size() = %d, from internal postion: %d\n", _buffer.size(), length, v.size(), _buffer_internal_position);
		if (_buffer_internal_position + __tmin(static_cast<long>(length), static_cast<long>(v.size())) > _max_buffer_length) {
			flush_buffer();
		}
		//cout << "buffer size allocated is " << _buffer.size() << " reading " << length << " positions, from internal postion: " << _buffer_internal_position << "\n";

		//errno_t err = memcpy_s(&_buffer[_buffer_internal_position], (_buffer.size() - _buffer_internal_position)*sizeof(T), &v[offset], length*sizeof(T));
		auto vstart = std::next(v.begin(), offset);
		auto vend = std::next(v.begin(), offset + length);
		auto bstart = std::next(_buffer.begin(), _buffer_internal_position);
		std::copy(vstart,vend ,bstart );
		/*
		BOOST_MY_THROW(err);
		cout << "from middle\n";
		for (int i = 0; i < 255; i++) {
		cout << _buffer[_buffer_internal_position + length / 2] << " ";
		if (i % 16 == 0 && i> 0) cout << "\n";
		}
		*/
		_buffer_internal_position += length;
		//std::cout << "copy " << length << " nodes to output buffer, buffer reached " << _buffer_internal_position << " internal position, for file name "<< _file_name<<"\n";
		//cout << "buffer size copied is " << length << " positions, internal position is now: " << _buffer_internal_position << "\n";
	}	// this overload from vector
	inline void close_file() { _file_handle.close_file(); }
};
template class OutputBuffer<double>;
template class OutputBuffer<float>;
#endif
