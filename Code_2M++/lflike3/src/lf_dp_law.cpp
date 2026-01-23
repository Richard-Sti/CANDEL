#include <boost/python.hpp>
#include <stdexcept>
#include "lf_dp_law.hpp"
#include "f77_mangling.hpp"
#include "error.hpp"

using boost::python::extract;

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

static const double g_log10 = log(10.0);



LuminosityFunction_DoublePowerLaw::LuminosityFunction_DoublePowerLaw()
{
  dplaw_alpha=-0.8;
  dplaw_beta=-1.2;
  dplaw_M0=-23.2;
  setMsun(3.29);
}

double LuminosityFunction_DoublePowerLaw::number_density(double M)
{
  double x = pow(10., 0.4*(dplaw_M0-M));
  if (x < 1) // Fainter than the limit
    return 0.4*g_log10*pow(x, 1+dplaw_alpha);
  else
    return 0.4*g_log10*pow(x, 1+dplaw_beta);	
}

double LuminosityFunction_DoublePowerLaw::integral_number_density(double M)
{
  double xfaint = pow(10., 0.4*(dplaw_M0-M));
  if (xfaint < 1) // Fainter than the knee luminosity
    {
      double part1 = 1/(1+dplaw_alpha) * (1-pow(xfaint, 1 + dplaw_alpha));
      double part2 = -1/(1+dplaw_beta);
      
      return part1+part2;
    }
  double part0 = -1/(1+dplaw_beta) * pow(xfaint, 1 + dplaw_beta);
  return part0;
}


double LuminosityFunction_DoublePowerLaw::integral_luminosity_density(double M)
{
  double xfaint = pow(10., 0.4*(dplaw_M0-M));
  if (xfaint < 1) // Fainter than the limit
    {
      double part1 = 1/(2+dplaw_alpha) * (1-pow(xfaint, 2 + dplaw_alpha));
      double part2 = -1/(2+dplaw_beta);
      
      return (part1+part2)*dplaw_L0;
    }
  double part0 = -1/(2+dplaw_beta) * dplaw_L0 * pow(xfaint, 2 + dplaw_beta);

  return part0;
}

void LuminosityFunction_DoublePowerLaw::setMsun(double Msun)
{
  this->Msun = Msun;
  dplaw_L0 = pow(10.0, 0.4*(Msun-dplaw_M0));
}

void LuminosityFunction_DoublePowerLaw::set_parameters(const boost::python::object& parameters)
{
  dplaw_M0 = extract<double>(parameters[0]);
  dplaw_alpha = extract<double>(parameters[1]);
  dplaw_beta = extract<double>(parameters[2]);
  dplaw_L0 = pow(10.0,0.4*(Msun-dplaw_M0));
}
