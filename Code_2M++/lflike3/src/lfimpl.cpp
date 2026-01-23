#include <boost/python.hpp>
#include <stdexcept>
#include "lfimpl.hpp"
#include "f77_mangling.hpp"
#include "error.hpp"

using boost::python::extract;

extern "C" double F77_FUNC(gamincc,GAMMINC)(double *a, double *x);

static double C_gamminc(double a, double x)
{
  return F77_FUNC(gamincc,GAMINCC)(&a, &x);
}

static const double g_log10 = log(10.0);


LuminosityFunctionImplementation::LuminosityFunctionImplementation()
{
}
 
LuminosityFunctionImplementation::~LuminosityFunctionImplementation()
{
}
