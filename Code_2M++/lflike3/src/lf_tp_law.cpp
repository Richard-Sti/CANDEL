#include <boost/python.hpp>
#include <stdexcept>
#include "lf_tp_law.hpp"
#include "f77_mangling.hpp"
#include "error.hpp"

using boost::python::extract;

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

static const double g_log10 = log(10.0);

LuminosityFunction_TriplePowerLaw::LuminosityFunction_TriplePowerLaw()
{
  dplaw_alpha=-0.8;
  dplaw_beta=-1.2;
  dplaw_M0=-23.2;
  setMsun(3.29);
}

double LuminosityFunction_TriplePowerLaw::number_density(double M)
{
  double x0 = pow(10., 0.4*(dplaw_M0-M));
  double x1 = pow(10., 0.4*(tplaw_M1-M));
  
  if (x0 < 1)
    {
      return 0.4*g_log10*pow(x0,1+dplaw_alpha);
    }
  else if (x1 < 1)
    {
      return 0.4*g_log10*pow(x0,1+dplaw_beta);
    }
  
  return 0.4*g_log10*pow(x0/x1,1+dplaw_beta)*pow(x1,1+tplaw_gamma);	
}

double LuminosityFunction_TriplePowerLaw::integral_number_density(double M)
{
  double x0 = pow(10., 0.4*(dplaw_M0-M));
  double x1 = pow(10., 0.4*(tplaw_M1-M));
  double r1 = x0/x1;
  double B = pow(r1, 1+dplaw_beta);
  
  if (x0 < 1) // Fainter than the knee luminosity
    {
      double part1 = 1/(1+dplaw_alpha) * (1-pow(x0, 1 + dplaw_alpha));
      double part2 = 1/(1+dplaw_beta) * (pow(r1,(1+dplaw_beta)) - 1);
      double part3 = -1/(1+tplaw_gamma) * B;
      
      return part1+part2+part3;
    }
  else if (x1 < 1)
    {
      double part1 = 1/(1+dplaw_beta) * (pow(r1,(1+dplaw_beta)) - pow(x0,1+dplaw_beta));
      double part2 = -1/(1+tplaw_gamma) * B;
      
      return part1+part2;
    }
  
  double part0 = -1/(1+tplaw_gamma) * B * pow(x1, 1 + tplaw_gamma);
  return part0;
}


double LuminosityFunction_TriplePowerLaw::integral_luminosity_density(double M)
{
  double x0 = pow(10., 0.4*(dplaw_M0-M));
  double x1 = pow(10., 0.4*(tplaw_M1-M));
  double r1 = x0/x1;
  double B = pow(r1, 1+dplaw_beta);
  if (x0 < 1) // Fainter than the knee luminosity
    {
      double part1 = 1/(2+dplaw_alpha) * (1-pow(x0, 2 + dplaw_alpha));
      double part2 = 1/(2+dplaw_beta) * (pow(r1,(2+dplaw_beta)) - 1);
      double part3 = 1/(2+tplaw_gamma) * B / r1;
      
      return dplaw_L0*(part1+part2+part3);
    }
  else if (x1 < 1)
    {
      double part1 = 1/(2+dplaw_beta) * (pow(r1,(2+dplaw_beta)) - pow(x0,2+dplaw_beta));
      double part2 = 1/(2+tplaw_gamma) * B / r1;
      
      return dplaw_L0*(part1 + part2);
    }
  
  double part0 = -1/(2+tplaw_gamma) * B * pow(x1, 2 + tplaw_gamma) / r1;
  return part0*dplaw_L0;
}


void LuminosityFunction_TriplePowerLaw::setMsun(double Msun)
{
  this->Msun = Msun;
  dplaw_L0 = pow(10.0, 0.4*(Msun-dplaw_M0));
}

void LuminosityFunction_TriplePowerLaw::set_parameters(const boost::python::object& parameters)
{
  dplaw_M0 =  extract<double>(parameters[0]);
  tplaw_M1 =  extract<double>(parameters[1]);
  dplaw_alpha = extract<double>(parameters[2]);
  dplaw_beta = extract<double>(parameters[3]);
  tplaw_gamma = extract<double>(parameters[4]);
  dplaw_L0 = pow(10.0,0.4*(Msun-dplaw_M0));
}
