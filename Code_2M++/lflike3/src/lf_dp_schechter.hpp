#ifndef __LF_DP_SCHECHTER_IMPL_HPP
#define __LF_DP_SCHECHTER_IMPL_HPP

#include "lfimpl.hpp"

class LuminosityFunction_DoubleSchechter: public LuminosityFunctionImplementation
{
public:
  double alpha, Mstar, alpha2, Mstar2, A;
  double Lstar;
  double Msun;

  LuminosityFunction_DoubleSchechter();

  void set_parameters(const boost::python::object& parameters);

  void setMsun(double M);

  double number_density(double M);
  double integral_number_density(double M);
  double integral_luminosity_density(double M);
};

#endif
