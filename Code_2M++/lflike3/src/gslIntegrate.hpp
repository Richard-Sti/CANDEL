#ifndef __MYGSL_INTEGRATE_HPP
#define __MYGSL_INTEGRATE_HPP

#include <boost/format.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <string>

class IntegrationException: virtual std::exception
{
public:
  IntegrationException(double a, double b)
   {
     s = boost::str(boost::format("Integration error with bounds [%g,%g]") % a % b);
   }

  virtual ~IntegrationException() throw() {}

  virtual  const char *what() const throw() { return s.c_str(); }

protected:
  std::string s;
};

template<typename FunT>
double gslSpecialFunction(double x, void *param)
{
  FunT *f = (FunT *)param;

  return (*f)(x);
}

template<typename FunT>
double gslIntegrate(FunT& v, double a, double b, double prec, double prec_abs = 0, int NPTS = 1024)
{
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(NPTS);
  gsl_function f;
  double result;
  double abserr;

  gsl_error_handler_t *errh = gsl_set_error_handler_off();

  f.function = &gslSpecialFunction<FunT>;
  f.params = &v;

  int rc = gsl_integration_qag(&f, a, b, prec_abs, prec, NPTS, GSL_INTEG_GAUSS61,
  	  	           w, &result, &abserr);
  
  gsl_integration_workspace_free(w);

  gsl_set_error_handler(errh);

  if (rc != 0)  
    throw IntegrationException(a,b);  

  return result;
}

#endif
