#include <boost/python.hpp>
#include <stdexcept>
#include "lf_dp_schechter.hpp"
#include "f77_mangling.hpp"
#include "error.hpp"

using boost::python::extract;

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

static const double g_log10 = log(10.0);



LuminosityFunction_DoubleSchechter::LuminosityFunction_DoubleSchechter()
{
  Mstar = -23.2;
  alpha = -1.1;
  Mstar2 = -23.;
  alpha2 = -1.1;
  A = 1;
  setMsun(3.29);
}

double LuminosityFunction_DoubleSchechter::number_density(double M)
{
  double x = pow(10., 0.4*(Mstar-M));
  double x1 = pow(10., 0.4*(Mstar2-M));	
  double r = pow(10., 0.4*(Mstar-Mstar2));
   
  return 0.4*g_log10*(  pow(x,(1+alpha))*exp(-x) + A*r*pow(x1,(1+alpha2))*exp(-x1));
}

double LuminosityFunction_DoubleSchechter::integral_number_density(double M)
{
  double x = pow(10., 0.4*(Mstar-M));
  double x2 = pow(10., 0.4*(Mstar2-M));
  double a = 1 + alpha;
  double a2 = 1+ alpha2;
  double r = pow(10., 0.4*(Mstar-Mstar2));

  return C_gamminc(a, x) + A * r * C_gamminc(a2, x2);
}

double LuminosityFunction_DoubleSchechter::integral_luminosity_density(double M)
{
  double x = pow(10., 0.4*(Mstar-M));
  double x2 = pow(10., 0.4*(Mstar2-M));
  double a = 2 + alpha;
  double a2 = 2 + alpha2;
  double r = pow(10., 0.4*(Mstar-Mstar2));

  return Lstar*(C_gamminc(a, x) + A*r*r * C_gamminc(a2, x2));  
}

void LuminosityFunction_DoubleSchechter::setMsun(double Msun)
{
  this->Msun = Msun;
  this->Lstar = pow(10.0, 0.4*(Msun-Mstar));
}


void LuminosityFunction_DoubleSchechter::set_parameters(const boost::python::object& parameters)
{
  Mstar = extract<double>(parameters[0]);
  alpha = extract<double>(parameters[1]);
  Mstar2 = extract<double>(parameters[2]);
  alpha2 = extract<double>(parameters[3]);
  A = extract<double>(parameters[4]);
  this->Lstar = pow(10.0, 0.4*(Msun-Mstar));
}
