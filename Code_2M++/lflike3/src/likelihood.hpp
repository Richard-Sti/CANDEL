#ifndef _LF_LIKELIHOOD_HPP
#define _LF_LIKELIHOOD_HPP

#include <boost/python.hpp>
#include <vector>

class LuminosityFunctionHelper;

class LuminosityFunctionLikelihood
{
public:
  LuminosityFunctionLikelihood(const boost::python::object& abs_mag, 
			       const boost::python::object& abs_maglimb, const boost::python::object& abs_maglimf,
			       const boost::python::object& cb, const boost::python::object& cf);
  ~LuminosityFunctionLikelihood();

  double computeLogLike(LuminosityFunctionHelper& helper);
  double computeLogOne(LuminosityFunctionHelper& helper, int i);

  double computeEffectiveNumber(LuminosityFunctionHelper& helper, double J3);

  void findBestParameters(LuminosityFunctionHelper& helper,
			  boost::python::object& parameters,
			  boost::python::object& step,
			  double eps_abs);

private:
  std::vector<double> abs_mag, abs_maglimb, abs_maglimf, cb, cf;
};

#endif
