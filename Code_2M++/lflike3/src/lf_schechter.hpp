#ifndef __LF_SCHECHTER_IMPL_HPP
#define __LF_SCHECHTER_IMPL_HPP

#include "lfimpl.hpp"

class LuminosityFunction_Schechter: public LuminosityFunctionImplementation
{
public:
  double alpha, Mstar;
  double Lstar;
  double Msun;

  LuminosityFunction_Schechter();

  void set_parameters(double mstar, double alpha);
  void set_parameters(const boost::python::object& parameters);

  void setMsun(double M);

  double number_density(double M);
  double integral_number_density(double M);
  double integral_luminosity_density(double M);
};

#endif
