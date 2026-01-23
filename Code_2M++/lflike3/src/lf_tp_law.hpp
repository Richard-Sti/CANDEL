#ifndef __LF_TP_LAW_IMPL_HPP
#define __LF_TP_LAW_IMPL_HPP

#include "lfimpl.hpp"

class LuminosityFunction_TriplePowerLaw: public LuminosityFunctionImplementation
{
public:
  double dplaw_M0, dplaw_alpha, dplaw_beta, dplaw_L0; 
  double Lstar;
  double Msun;
  double tplaw_gamma, tplaw_M1; // LF_TRIPLE_POWER_LAW_FUNCTION support

  LuminosityFunction_TriplePowerLaw();

  void set_parameters(const boost::python::object& parameters);

  void setMsun(double M);

  double number_density(double M);
  double integral_number_density(double M);
  double integral_luminosity_density(double M);
};

#endif
