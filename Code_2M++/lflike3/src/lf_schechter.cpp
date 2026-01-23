#include <boost/python.hpp>
#include <stdexcept>
#include "lf_schechter.hpp"
#include "f77_mangling.hpp"
#include "error.hpp"

using boost::python::extract;

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

static const double g_log10 = log(10.0);



LuminosityFunction_Schechter::LuminosityFunction_Schechter()
{
  Mstar = -23.2;
  alpha = -1.1;
  Msun = 3.29;
}

double LuminosityFunction_Schechter::number_density(double M)
{
  double x = pow(10.0, 0.4*(Mstar-M));
  return 0.4*g_log10*pow(x,(1+alpha))*exp(-x);
}

double LuminosityFunction_Schechter::integral_number_density(double M)
{
  double xfaint = pow(10., 0.4*(Mstar-M));
  double a = 1 + alpha;
  
  return C_gamminc(a, xfaint);
}

double LuminosityFunction_Schechter::integral_luminosity_density(double M)
{
  double xfaint = pow(10., 0.4*(Mstar-M));
  double a = 2 + alpha;
  return Lstar*C_gamminc(a, xfaint);
}

void LuminosityFunction_Schechter::setMsun(double Msun)
{
  this->Msun = Msun;
  this->Lstar = pow(10.0, 0.4*(Msun-Mstar));
}

void LuminosityFunction_Schechter::set_parameters(double mstar, double alpha)
{
  this->Mstar = mstar;
  this->alpha = alpha;
  this->Lstar = pow(10.0, 0.4*(Msun-Mstar));
}

void LuminosityFunction_Schechter::set_parameters(const boost::python::object& parameters)
{
  double a = extract<double>(parameters[0]);
  double b = extract<double>(parameters[1]);
  set_parameters(a, b);
}
