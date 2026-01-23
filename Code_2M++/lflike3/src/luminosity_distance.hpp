#ifndef __LUMINOSITY_DISTANCE_HPP
#define __LUMINOSITY_DISTANCE_HPP

#include <cmath>
#include <vector>
#include "cosmo.hpp"
#include "gslIntegrate.hpp"
#include "interpolate.hpp"

struct DL_Integrator
{
  Cosmo cosmos;
  double operator()(double z)
  {
    double om0 = cosmos.omega_m;
    double ode0 = cosmos.omega_de;
    double ok0 = 1- om0 - ode0;
    double w = cosmos.w;
    double or0 = 0;

    double a = 1/(1+z);
    double a2 = a*a;
    double a3 = a2*a;
    double a4 = a2*a2;
    
    return 1/std::sqrt(om0/a3+ok0/a2+or0/a4 + ode0/std::pow(a, 3*(1+w)));
  }
};

class Cosmo;

class LuminosityDistance
{
private:
  LF_Interpolate dl_interpolate;
public:
  LuminosityDistance(const Cosmo& cosmo)
  {
    this->cosmos = cosmo;
    nw = 501;
    xa.resize(nw);
    za.resize(nw);

    setup_dl();
    
    dl_interpolate = LF_Interpolate(&xa[0], &za[0], nw);  
  }

  ~LuminosityDistance()
  {
  }

  double dl(double z)
  {
    double x = -std::log(1+z);

    return dl_interpolate.compute(x);
  }

  void setup_dl()
  {
    double omega_k = 1 - cosmos.omega_m - cosmos.omega_de;
    double errtol = 1e-7;
    DL_Integrator dl_one_over_h;

    dl_one_over_h.cosmos = cosmos;
    std::cout << "Setup_Dl" << std::endl;

    for (int i = 0; i < nw-1; i++)
      {
	double x = (i-nw+1)*0.01;
	double eta;

	xa[i] = x;
        try
         {
	    eta = gslIntegrate(dl_one_over_h, 0, std::exp(-x)-1, errtol);
         } 
         catch(const IntegrationException& e)
        {
          std::cout << boost::format("Error while integrating 1/H: i = %d, x = %g, Omega_M = %g, Omega_DE = %g") % i % x % cosmos.omega_m % cosmos.omega_de << std::endl;
          throw e;
        }

	if (fabs(omega_k) < 1e-5)
	  za[i] = 2998. * std::exp(-x) * eta;
	else if (omega_k > 0)
	  za[i] = 2998. * std::exp(-x) * std::sinh(std::sqrt(omega_k)*eta)/std::sqrt(omega_k);
	else 
	  za[i] = 2998. * std::exp(-x) * std::sin(std::sqrt(-omega_k)*eta)/std::sqrt(-omega_k);
      }
    std::cout << "Finish_Dl" << std::endl;
    xa[nw-1] = 0;
    za[nw-1] = 0;
  }

private:
  Cosmo cosmos;
  int nw;
  std::vector<double> xa, za;
};

#endif
