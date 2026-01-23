#ifndef _LF_HELPER_HPP
#define _LF_HELPER_HPP

#include <iostream>
#include <boost/python.hpp>
#include <vector>
#include "luminosity_distance.hpp"
#include "lfimpl.hpp"

enum WeighingScheme
  {
    USE_NUMBER_SCHEME=0,
    USE_LUMINOSITY_SCHEME=1,
    USE_HOMOGENEOUS_LUMINOSITY_SCHEME=2
  };

extern boost::python::object global_numpy;

class LuminosityFunctionWrapper : public boost::python::object, public LuminosityFunctionImplementation
{
public:
  LuminosityFunctionWrapper()
  {
  }

  double number_density(double M)
  {
    using boost::python::extract;
    return extract<double>(attr("number_density")(M));
  }

  double integral_number_density(double M)
  {
    using boost::python::extract;
    return extract<double>(attr("integral_number_density")(M));
  }

  double integral_luminosity_density(double M)
  {
    using boost::python::extract;
    return extract<double>(attr("integral_luminosity_density")(M));
  }

  void setMsun(double M)
  {
    attr("setMsun")(M);
  }

  void set_parameters(const boost::python::object& parameters)
  {
    using boost::python::extract;
    attr("set_parameters")(parameters);
  }
};

class LuminosityFunctionHelper
{
public:
  enum LF_Type
    {
      LF_PYTHON_FUNCTION,
      LF_SCHECHTER_FUNCTION,
      LF_DOUBLE_POWER_LAW_FUNCTION,
      LF_TRIPLE_POWER_LAW_FUNCTION,
      LF_DOUBLE_SCHECHTER_FUNCTION,
      LF_MULTI_SCHECHTER_FUNCTION,
    };

  LF_Type lf_type;
  LuminosityFunctionWrapper lf_object; // Support for LF_PYTHON_FUNCTION
  LuminosityFunctionImplementation *lfimpl; // Generic luminosity function implementation

public:
  LuminosityFunctionHelper() {
    abs_maglimb = -30;
    abs_maglimf = -10;
    eps_abs = 0.;
    eps_rel = 1e-3;
    Msun = 3.29; // Note: in K-band for 2MASS.

    weight_scheme = USE_NUMBER_SCHEME;
    
    lfimpl = 0;
    setLuminosityFunction(LF_SCHECHTER_FUNCTION);
  }

  ~LuminosityFunctionHelper() {}

  void setSunAbsoluteMagnitude(double Msun)
  {
    this->Msun = Msun;
    this->lfimpl->setMsun(Msun);
  }

  void setCosmology(Cosmo& cosmos)
  {
    this->cosmos = cosmos;
  }

  void setLuminosityFunction(LF_Type lf_type);

  void updateLuminosityFunctionParameters(const boost::python::object&  parameters);

  void setLuminosityFunction_custom(boost::python::object& lf)
  {
    using boost::python::extract;

    double r;
    // Attempt to call the functor just in case it throws something.
    std::cout << "Attempting to call specified function" << std::endl;
    r = extract<double>(lf.attr("number_density")(abs_maglimb));
    r = extract<double>(lf.attr("integral_number_density")(abs_maglimb));
    r = extract<double>(lf.attr("integral_luminosity_density")(abs_maglimb));
    
    lf_type = LF_PYTHON_FUNCTION;
    lf_object = extract<LuminosityFunctionWrapper>(lf);
  }

  void set_absmag_range(double abs_maglimb, double abs_maglimf)
  {
    this->abs_maglimb = abs_maglimb;
    this->abs_maglimf = abs_maglimf;
  }

  void set_precision(double eps_abs, double eps_rel)
  {
    this->eps_abs = eps_abs;
    this->eps_rel = eps_rel;
  }

  double computeDavisHuchraNormalization_magrange(double d0_start, double d0_end,
						  double app_maglimf, double app_maglimb,
						  boost::python::object cb, boost::python::object cf, double J3);

  double computePhiStarNormalisation
     (double maglimf, double maglimb, boost::python::object cf, boost::python::object cb,
      double dmin, double dmax, double J3, double epsrel);
    
  boost::python::object computeAbsoluteMagnitudes(boost::python::object &distances, boost::python::object& apparent_mag);

  double integral_nw_lumfun(double Mfaint);
  double integral_nw_lumfun_2m(double Mfaint, double Mbright);

  double lumfun(double M);
  double log_lumfun(double M);

  double integral_lw_lumfun(double Mfaint);
  double integral_lw_lumfun_2m(double Mfaint, double Mbright);
  
  // scheme = 0 => number weighing, 1 => luminosity weighing
  void chooseWeight(int scheme)
  {
    weight_scheme = scheme;
  }

  // The result depends on the selected scheme (number or luminosity).
  double getWeight_1m(double M);
  double getWeight_2m(double Mf, double Mb, double cf, double cb);

  double getMaximumMagnitude() const { return abs_maglimf; }
  double getMinimumMagnitude() const { return abs_maglimb; }

  void computeGalaxyWeights(const boost::python::object& appmag,
			    const boost::python::object& appmag_limf,
			    const boost::python::object& appmag_limb,
			    const boost::python::object& cb,
			    const boost::python::object& cf,
			    const boost::python::object& redshifts,
			    boost::python::object& weights);

private:
  double abs_maglimb, abs_maglimf;
  double eps_abs, eps_rel;
  double Msun;
  Cosmo cosmos;
  int weight_scheme;

  friend struct Integrand_DavisHuchra_singleC;
  friend struct Integrand_DavisHuchra;
};


#endif
