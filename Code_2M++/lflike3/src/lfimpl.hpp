#ifndef __LF_IMPL_HPP
#define __LF_IMPL_HPP

#include <boost/python.hpp>

class LuminosityFunctionImplementation
{
public:
  LuminosityFunctionImplementation();
  virtual ~LuminosityFunctionImplementation();

  virtual double number_density(double M) = 0;
  virtual double integral_number_density(double M) = 0;
  virtual double integral_luminosity_density(double M) = 0;
  virtual void setMsun(double M) = 0;
  
  virtual void set_parameters(const boost::python::object& parameters) = 0;
};

#endif
