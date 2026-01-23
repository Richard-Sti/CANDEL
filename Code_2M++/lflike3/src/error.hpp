#ifndef _LFLIKE3_ERRORS_HPP
#define _LFLIKE3_ERRORS_HPP

#include <exception>
#include <string>

class LuminosityFunctionException: public std::exception
{
public:
  LuminosityFunctionException(const std::string& err) : msg(err) {}
  ~LuminosityFunctionException() throw() {}

  char const * what() const throw() { return msg.c_str(); }

private:
  std::string msg;
};

#endif
