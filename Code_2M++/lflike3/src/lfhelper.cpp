#include <boost/math/special_functions/gamma.hpp>
#include <gsl/gsl_sf.h>
#include <cmath>
#include <iostream>
#include <boost/python.hpp>
#include <boost/format.hpp>
#include <string>
#include "luminosity_distance.hpp"
#include "lfhelper.hpp"
#include "f77_mangling.hpp"
#include <stdexcept>
#include "error.hpp"
#include "lfimpl.hpp"
#include "lf_schechter.hpp"
#include "lf_dp_schechter.hpp"
#include "lf_dp_law.hpp"
#include "lf_tp_law.hpp"

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

using std::log10;
using std::cout;
using std::endl;
using std::min;
using std::max;

using boost::format;
using boost::python::extract;
using boost::python::object;
//using boost::math::tgamma;

//#define gsl_sf_gamma_inc(a,z) tgamma(a,z)

#define LIGHT_SPEED 299792.458

object global_numpy;

static double cube(double a)
{
  return a*a*a;
}

double LuminosityFunctionHelper::getWeight_1m(double M)
{
  double numer, denom;

  if (M < abs_maglimb || M > abs_maglimf)
    {
      throw LuminosityFunctionException(str(format("Illegal magnitude in numprob M=%g") % M));
    }

  switch (weight_scheme)
    {
    case USE_NUMBER_SCHEME:
      numer = integral_nw_lumfun_2m(M, abs_maglimb);
      denom = integral_nw_lumfun_2m(abs_maglimf, abs_maglimb);
      break;
    case USE_LUMINOSITY_SCHEME:      
      numer = integral_lw_lumfun_2m(M, abs_maglimb);
      denom =  integral_lw_lumfun_2m(abs_maglimf, abs_maglimb);
      break;
    case USE_HOMOGENEOUS_LUMINOSITY_SCHEME:
      denom = 1;
      numer = integral_lw_lumfun_2m(abs_maglimf, abs_maglimb);
      break;
    default:
      throw LuminosityFunctionException("Illegal scheme");
    }

  return numer/denom;
}

double LuminosityFunctionHelper::getWeight_2m(double Mf, double Mb, double cf, double cb)
{
  double numer, denom;

  if (Mb < abs_maglimb)
    {
      throw LuminosityFunctionException(str(format("Invalid bright magnitude mb=%g < Mb=%g") % Mb % abs_maglimb));
    }

  if (Mf > abs_maglimf)
    {
      throw LuminosityFunctionException(str(format("Invalid faint magnitude magnitude mf=%g > Mf=%g") % Mf % abs_maglimf));
    }
  switch (weight_scheme)
    {
    case USE_NUMBER_SCHEME:
      numer = cf * integral_nw_lumfun_2m(Mf, Mb)  + cb * integral_nw_lumfun_2m(Mb, abs_maglimb);
      denom = integral_nw_lumfun_2m(abs_maglimf, abs_maglimb);
      break;
    case USE_LUMINOSITY_SCHEME:
      numer = cf * integral_lw_lumfun_2m(Mf, Mb)  + cb * integral_lw_lumfun_2m(Mb, abs_maglimb);
      denom = integral_lw_lumfun_2m(abs_maglimf, abs_maglimb);
      break;
    case USE_HOMOGENEOUS_LUMINOSITY_SCHEME:
      numer = integral_lw_lumfun_2m(abs_maglimf, abs_maglimb);
      denom = 1;
      break;
    default:
      throw LuminosityFunctionException("Invalid weighing scheme");
    }
     
  return numer/denom;
}

struct Integrand_DavisHuchra
{
  LuminosityFunctionHelper *helper;
  double cb, cf;
  double app_maglimb, app_maglimf;
  double J3;
  
  double operator()(double d)
  {
    double d_modulus = 5.0*log10(d*1e5);

    double mag_bright = app_maglimb - d_modulus;
    double mag_faint = app_maglimf - d_modulus;
    double w;
    
    mag_faint = min(mag_faint,  helper->abs_maglimf);
    mag_bright = min(mag_bright, helper->abs_maglimf);
    mag_bright = max(mag_bright, helper->abs_maglimb);
    mag_faint = max(mag_bright, mag_faint);

    w = helper->getWeight_2m(mag_faint, mag_bright, cf, cb);

    return d*d * w / (1 + J3 * w);
  }
};

struct Integrand_DavisHuchra_singleC
{
  LuminosityFunctionHelper *helper;
  double c;
  double app_maglimb;
  double J3;
  
  double operator()(double d)
  {
    double d_modulus = 5.0*log10(d*1e5);

    double mag_bright = app_maglimb - d_modulus;
    double w;
    
    mag_bright = min(mag_bright, helper->abs_maglimf);
    mag_bright = max(mag_bright, helper->abs_maglimb);

    w = helper->getWeight_1m(mag_bright) * c;

    return d*d * w / (1 + J3 * w);
  }
};

double 
  LuminosityFunctionHelper::computeDavisHuchraNormalization_magrange
            (double d0_start, double d0_end, double app_maglimf, double app_maglimb,
	     object cf, object cb, double J3)
{
  using boost::python::len;
  int Ncb = len(cb);
  int Ncf = len(cf);
  double result, final_result;
  Integrand_DavisHuchra weights;
  Integrand_DavisHuchra_singleC weights_singleC;

  if (Ncb != Ncf)
    return -1;

  final_result = 0;

  weights.helper = this;
  weights.J3 = J3;
  weights.app_maglimb = app_maglimb;
  weights.app_maglimf = app_maglimf;

  weights_singleC.helper = this;
  weights_singleC.J3 = J3;
  weights_singleC.app_maglimb = app_maglimb;

  for (int i = 0; i < Ncb; i++)
    {
      double local_cb = extract<double>(cb[i]);
      double local_cf = extract<double>(cf[i]);

      if (local_cf <= 0)
	{
	  // This is the distance where the catalog is volume limited
	  // and thus the completeness is "1"
	  double d0_numstart = min(pow(10, 0.2*(app_maglimb - abs_maglimf - 25)), d0_end);
	  // This is the distance where nothing is left. So we should cut the integral
	  // to avoid convergence issues
	  double d0_numend = min(pow(10, 0.2*(app_maglimb - abs_maglimb - 25)), d0_end);

	  weights_singleC.c = local_cb;

	  cout << format("d0_numstart=%g d0_numend=%g d0_start=%g") % d0_numstart  % d0_numend % d0_start << endl;

	  if (d0_numstart > d0_start)
	    {
	      result = gslIntegrate(weights_singleC, d0_numstart, d0_numend, eps_rel, eps_abs);
	      result += local_cb * (cube(d0_numstart)-cube(d0_start))/3/(1+J3);
	    }
	  else
	    {
	      result = gslIntegrate(weights_singleC, d0_start, d0_numend, eps_rel, eps_abs);
	    }
	  cout << format("result = %g") % result << endl;
	}
      else if (local_cb <= 0)
	{
	  // Compute again when the integral is interesting. In that case, when it is non-null
	  // d0_numend (resp. d0_numstart) holds the farther (nearest) distance where the integrand
	  // is non-null

	  double d0_numend = pow(10, 0.2*(app_maglimf - abs_maglimb - 25));
	  double d0_numstart = pow(10, 0.2*(app_maglimb - abs_maglimf - 25));

	  d0_numend = min(d0_end, d0_numend);
	  d0_numstart = min(d0_start, d0_numstart);

	  cout << format("[cb<=0] d0_numstart=%g d0_numend=%g d0_start=%g") % d0_numstart  % d0_numend % d0_start << endl; 
	  cout << format("cb=%g cf=%g") %  local_cb % local_cf << endl;

	  if (d0_numstart < d0_numend)
	    {
	      weights.cb = local_cb;
	      weights.cf = local_cf;

	      result = gslIntegrate(weights, d0_numstart, d0_numend, eps_rel, eps_abs);
	      cout << format("Got result=%g") % result << endl;
	    }
	  else
	    {
	      result = 0;
	    }
	}
      else
	{
	  weights.cb = local_cb;
	  weights.cf = local_cf;

	  result = gslIntegrate(weights, d0_start, d0_end, eps_rel, eps_abs);
	}
      final_result += result;
    }

  cout << format("final_result = %g") % final_result << endl;

  return final_result;
}

double LuminosityFunctionHelper::computePhiStarNormalisation
        (double maglimf, double maglimb, object cf, object cb,
	 double dmin, double dmax, double J3, double epsrel)
{ 
  LuminosityDistance dl(cosmos);
  double drealMin = dl.dl(dmin/LIGHT_SPEED);
  double drealMax = dl.dl(dmax/LIGHT_SPEED);
  double epsabs;

  if (J3==0.0)
    epsabs = (cube(drealMax) - cube(drealMin))/3.*epsrel;      
  else
    epsabs = (cube(drealMax) - cube(drealMin))/3.*epsrel/J3;

  return computeDavisHuchraNormalization_magrange(drealMin, drealMax, maglimf, maglimb, cf, cb, J3);
}

// ===============================================================================================
// Number weighed integral of the luminosity function

double LuminosityFunctionHelper::integral_nw_lumfun(double Mfaint)
{
  switch (lf_type)
    {
    case LF_TRIPLE_POWER_LAW_FUNCTION:      
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_SCHECHTER_FUNCTION:
    case LF_PYTHON_FUNCTION:
      return lfimpl->integral_number_density(Mfaint);
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}

double LuminosityFunctionHelper::integral_nw_lumfun_2m(double Mfaint, double Mbright)
{
  static double max_time = 0;
  struct timeval tv, tv2;

  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
    case LF_PYTHON_FUNCTION:
      {
	return integral_nw_lumfun(Mfaint)-integral_nw_lumfun(Mbright);
      }
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}

// ===============================================================================================
// Luminosity weighed integral of the luminosity function

double LuminosityFunctionHelper::integral_lw_lumfun(double Mfaint)
{
  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_PYTHON_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
      return lfimpl->integral_luminosity_density(Mfaint);
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}

double LuminosityFunctionHelper::integral_lw_lumfun_2m(double Mfaint, double Mbright)
{
  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
    case LF_PYTHON_FUNCTION:
      return integral_lw_lumfun(Mfaint)-integral_lw_lumfun(Mbright);
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}

object LuminosityFunctionHelper::computeAbsoluteMagnitudes(object& distances, object& apparent_magnitude)
{
  using boost::python::len;
  using boost::python::exec;
  using boost::python::make_tuple;
  using std::log10;

#define E(a) extract<double>((a).attr("__float__")())

  int size = len(distances);
  object absmag_object = global_numpy.attr("empty")(make_tuple(size));
  LuminosityDistance dl(cosmos);

  for (int i = 0; i < size; i++)
    {
      double d = dl.dl(E(distances[i])*100/LIGHT_SPEED);
      double mu = 5.0*log10(d*1e5);

      absmag_object[i] = apparent_magnitude[i] - mu;
    }
#undef E

  return absmag_object;
}


double LuminosityFunctionHelper::lumfun(double M)
{
  static const double g_log10 = log(10.0);

  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_PYTHON_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
     return lfimpl->number_density(M);

    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }  
}

double LuminosityFunctionHelper::log_lumfun(double M)
{
  static const double g_log = log(0.4*log(10.0));
  static const double g_log10 = log(10.0);

  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_PYTHON_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
      return log(lfimpl->number_density(M));
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }  
}

void LuminosityFunctionHelper::updateLuminosityFunctionParameters(const boost::python::object&  parameters)
{
  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
    case LF_DOUBLE_SCHECHTER_FUNCTION:
    case LF_PYTHON_FUNCTION:
    case LF_DOUBLE_POWER_LAW_FUNCTION:
    case LF_TRIPLE_POWER_LAW_FUNCTION:
     lfimpl->set_parameters(parameters);
     break;
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}

void LuminosityFunctionHelper::computeGalaxyWeights(const object& appmag,
						    const object& appmag_limf,
						    const object& appmag_limb,
						    const object& all_cf,
						    const object& all_cb,
						    const object& redshifts,
						    object& weights)
{
  int N = len(appmag);
  LuminosityDistance dl(cosmos);
  using boost::python::extract;
  using boost::python::len;

#define E(a) extract<double>((a).attr("__float__")())

  if (len(appmag_limf) != N || len(appmag_limb) != N ||
      len(all_cb) != N || len(all_cf) != N  || len(weights) != N ||
      len(redshifts) != N)
    throw std::invalid_argument("All arrays must have the same size.");

  for (int i = 0; i < N; i++)
    {
      double m = E(appmag[i]), mb = E(appmag_limb[i]), mf = E(appmag_limf[i]),
	cb = E(all_cb[i]), cf = E(all_cf[i]);
      double z = std::max( E(redshifts[i])/LIGHT_SPEED,  100./LIGHT_SPEED); // Impose 1 Mpc/h threshold for distance determination.kk
      double d = dl.dl(z);
      double mu = 5*std::log10(d*1e5);
      double M = m-mu, Mb = mb-mu, Mf = mf-mu;

      Mb = min(max(Mb, abs_maglimb), abs_maglimf);
      Mf = min(max(Mf, abs_maglimb), abs_maglimf);
      if (M >= abs_maglimb && M <= abs_maglimf && M <= Mf) 
	weights[i] = 1.0/getWeight_2m(Mf, Mb, cf, cb);
      else
	weights[i] = 0; // This does not belong to the catalog     
    }
#undef E
}

void LuminosityFunctionHelper::setLuminosityFunction(LF_Type lf_type)
{
  if (lfimpl != 0 && lfimpl != &lf_object)
    {
      delete lfimpl;
      lfimpl = 0;
    }

  this->lf_type = lf_type;
  switch (lf_type)
    {
    case LF_SCHECHTER_FUNCTION:
      lfimpl = new LuminosityFunction_Schechter();
      break;
    case LF_DOUBLE_SCHECHTER_FUNCTION:
      lfimpl = new LuminosityFunction_DoubleSchechter();
      break;
    case LF_DOUBLE_POWER_LAW_FUNCTION:
      lfimpl = new LuminosityFunction_DoublePowerLaw();
      break;
    case LF_TRIPLE_POWER_LAW_FUNCTION:
      lfimpl = new LuminosityFunction_TriplePowerLaw();
      break;
    case LF_PYTHON_FUNCTION:
      lfimpl = &lf_object;
      break;
    default:
      throw LuminosityFunctionException("Unknown luminosity function");
    }
}
