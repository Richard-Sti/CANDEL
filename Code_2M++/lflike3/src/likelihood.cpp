#include <sys/time.h>
#include <cmath>
#include <gsl/gsl_multimin.h>
#include <stdexcept>
#include <boost/python.hpp>
#include "likelihood.hpp"
#include <boost/format.hpp>
#include "cosmo.hpp"
#include "lfhelper.hpp"
#include "error.hpp"

using boost::python::object;

using std::min;
using std::max;
using std::log;
using boost::bind;
using std::cout;
using std::endl;
using boost::format;

LuminosityFunctionLikelihood::LuminosityFunctionLikelihood(const object& absmag, 
							   const object& absmaglimb, const object& absmaglimf,
							   const object& cb, const object& cf)
{
  using boost::python::len;
  using boost::python::extract;

  int N = len(absmag);

  if (len(absmaglimb) != N || len(absmaglimf) != N ||
      len(cb) != N || len(cf) != N)
    throw std::invalid_argument("All arrays must be iterable and have the same size.");

  this->abs_mag.resize(N);
  this->abs_maglimb.resize(N);
  this->abs_maglimf.resize(N);
  this->cb.resize(N);
  this->cf.resize(N);

  for (int i = 0; i < N; i++)
    {
      abs_mag[i] = extract<double>(absmag[i]);
      abs_maglimb[i] = extract<double>(absmaglimb[i]);
      abs_maglimf[i] = extract<double>(absmaglimf[i]);
      this->cb[i] = extract<double>(cb[i]);
      this->cf[i] = extract<double>(cf[i]);
    }
}

LuminosityFunctionLikelihood::~LuminosityFunctionLikelihood()
{
}

struct LogLikelihood_Walker
{
  LuminosityFunctionHelper *helper;
  boost::python::object current;
  int N;
  LuminosityFunctionLikelihood *likelihood;

  double compute(const gsl_vector *x)
  {
    double loglike;

    // Transform to array
    cout << "Parameters: ";
    for (int i = 0; i < N; i++)
      {
	cout << gsl_vector_get(x, i) << " ";
	current[i] = gsl_vector_get(x, i);
      }
    cout << endl;

    helper->updateLuminosityFunctionParameters(current);
    loglike =  likelihood->computeLogLike(*helper);
    return loglike;
  }
};

static double multimin_adaptor(const gsl_vector * x, void *params)
{
  LogLikelihood_Walker *walker = (LogLikelihood_Walker *)params;

  return walker->compute(x);
}

void LuminosityFunctionLikelihood::findBestParameters(LuminosityFunctionHelper& helper,
						      boost::python::object& parameters,
						      boost::python::object& step,
						      double eps_abs)
{
  using boost::python::len;
  using boost::python::extract;

  int N = len(parameters);

  if (len(step) != N)
    {
      throw std::invalid_argument("Parameters and Step should be same size array");
    }

  gsl_multimin_fminimizer *fmin = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, N);
  gsl_multimin_function lf_fun;
  gsl_vector *x = gsl_vector_alloc(N);
  gsl_vector *step_x = gsl_vector_alloc(N);
  LogLikelihood_Walker walker;

  lf_fun.f = &multimin_adaptor;
  lf_fun.params = &walker;
  lf_fun.n = N;

  walker.likelihood = this;
  walker.helper = &helper;
  walker.current = parameters;
  walker.N = N;

  cout << "Loading parameters in minimizer" << endl;

  try
    {
      for (int i = 0; i < N; i++)
	{
	  cout << "Param " << i << endl;
	  gsl_vector_set(x, i, extract<double>(parameters[i]));
	  cout << "Step " << i << endl;
	  gsl_vector_set(step_x, i, extract<double>(step[i]));	  
	}
      
      gsl_multimin_fminimizer_set(fmin, &lf_fun, x, step_x);
    }
  catch(const std::exception& e)
    {
      gsl_vector_free(x);
      gsl_vector_free(step_x);
      gsl_multimin_fminimizer_free(fmin);
      throw;
    }

  
  gsl_vector_free(x);
  gsl_vector_free(step_x);
  
  
  cout << "Run miminization" << endl;
  try
    {
      do
	{	  
	  gsl_multimin_fminimizer_iterate(fmin);
	  //	  cout << format("    iterated (size=%g)") % gsl_multimin_fminimizer_size(fmin) << endl;
	}
      while (gsl_multimin_test_size(gsl_multimin_fminimizer_size(fmin), eps_abs));
      
      //      cout << "Reached convergence. Store the result." << endl;
      gsl_vector *current = gsl_multimin_fminimizer_x (fmin);
      for (int i = 0; i < N; i++)
	{
	  parameters[i] = gsl_vector_get(current, i);
	}
    }
  catch (const std::exception& e)
    {
      gsl_multimin_fminimizer_free(fmin);
      throw;
    }

  //  cout << "Freeing and returning." << endl;
  gsl_multimin_fminimizer_free(fmin);
}

double LuminosityFunctionLikelihood::computeLogLike(LuminosityFunctionHelper& helper)
{
  double total_log = 0;
  struct timeval tv, tv2;

  gettimeofday(&tv, 0);

  for (int i = 0 ; i < abs_mag.size(); i++)
    {
      double delta_log;
      delta_log = computeLogOne(helper, i);
      if (isnan(delta_log))
	throw LuminosityFunctionException("NaN detected in likelihood evaluation");
      if (delta_log > 1e10)
	throw LuminosityFunctionException(str(format("Extremely low likelihood for %d . Stopping") % i));
      total_log += delta_log;
    }

  gettimeofday(&tv2, 0);

  double delta = double(tv2.tv_sec-tv.tv_sec)*1000 + (double(tv2.tv_usec)-tv.tv_usec)/1000.;
  return -2.0*total_log;
}

double LuminosityFunctionLikelihood::computeLogOne(LuminosityFunctionHelper& helper, int i)
{
  double M = abs_mag[i];
  double Mb = abs_maglimb[i];
  double Mf = abs_maglimf[i];
  double c_bright = cb[i];
  double c_faint = cf[i];
  double log_numer, denom;

  log_numer = helper.log_lumfun(M);

  if (M < Mb)
    {
      log_numer += log(c_bright);
    }
  else if (M < Mf)
    {
      log_numer += log(c_faint);
    }
  else
    abort();

  double Brightest = helper.getMinimumMagnitude();
  double Faintest = helper.getMaximumMagnitude();

  if (M < Brightest || M > Faintest)
    return 0;

  double real_min_mag = max(min(Faintest, Mb), Brightest);
  double real_max_mag = max(min(Faintest, Mf), Brightest);

  assert(real_max_mag >= real_min_mag);

  denom = c_bright * helper.integral_nw_lumfun_2m(real_min_mag, Brightest)
    + c_faint * helper.integral_nw_lumfun_2m(real_max_mag, real_min_mag);

  if (denom <= 0)
    {
      throw LuminosityFunctionException(str(format("Denominator in the likelihood is null or negative (denom=%g). Stopping") % denom));
    }
  assert(denom > 0);

  return log_numer - log(denom);
}


double  LuminosityFunctionLikelihood::computeEffectiveNumber(LuminosityFunctionHelper& helper, double J3)
{
  double N = 0;
  double Brightest = helper.getMinimumMagnitude();
  double Faintest = helper.getMaximumMagnitude();
  double EffectiveNumber = 0;

  double denom = helper.integral_nw_lumfun_2m(Faintest, Brightest);

  for (int i = 0; i < abs_mag.size(); i++)
    {
      double M = abs_mag[i];
      double Mb = abs_maglimb[i];
      double Mf = abs_maglimf[i];
      double c_bright = cb[i];
      double c_faint = cf[i];

      double real_bright_mag = max(min(Faintest, Mb), Brightest);
      double real_faint_mag = max(min(Faintest, Mf), Brightest);
      double numer, w;

      numer = c_faint * helper.integral_nw_lumfun_2m(real_faint_mag, real_bright_mag) + 
	c_bright * helper.integral_nw_lumfun_2m(real_bright_mag, Brightest);
      w = numer / denom;
      w = 1.0/(1.0+J3*w);
      EffectiveNumber += w;
    }

  return EffectiveNumber;
}
