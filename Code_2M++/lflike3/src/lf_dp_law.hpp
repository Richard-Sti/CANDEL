#ifndef __LF_DP_LAW_IMPL_HPP
#define __LF_DP_LAW_IMPL_HPP

#include "lfimpl.hpp"

class LuminosityFunction_DoublePowerLaw: public LuminosityFunctionImplementation
{
public:
  double dplaw_M0, dplaw_alpha, dplaw_beta, dplaw_L0; 
  double Lstar;
  double Msun;

  LuminosityFunction_DoublePowerLaw();

  void set_parameters(const boost::python::object& parameters);

  void setMsun(double M);

  double number_density(double M);
  double integral_number_density(double M);
  double integral_luminosity_density(double M);
};

#endif
