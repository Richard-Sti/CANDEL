#include "gslIntegrate.hpp"
#include <boost/python.hpp>
#include <boost/format.hpp>
#include "lfhelper.hpp"
#include "likelihood.hpp"
#include "lfimpl.hpp"
#include "error.hpp"

static void translate_to_python(LuminosityFunctionException const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

static void translate_int_to_python(IntegrationException const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE(_lflike)
{
  using namespace boost::python;

  global_numpy = import("numpy");

  register_exception_translator<LuminosityFunctionException>(&translate_to_python);
  register_exception_translator<IntegrationException>(&translate_int_to_python);

  class_<LuminosityFunctionImplementation, boost::noncopyable>("LuminosityFunctionImplementation", no_init)
    .def("number_density", &LuminosityFunctionImplementation::number_density, args("M"))
    .def("integral_number_density", &LuminosityFunctionImplementation::integral_number_density, args("M"))
    .def("integral_luminosity_density", &LuminosityFunctionImplementation::integral_luminosity_density, args("M"))
    .def("setMsun", &LuminosityFunctionImplementation::setMsun, args("M"))
    .def("set_parameters", &LuminosityFunctionImplementation::set_parameters);

  class_<LuminosityFunctionHelper>("LuminosityFunctionHelper")
    .def("set_absmag_range", &LuminosityFunctionHelper::set_absmag_range)
    .def("set_precision", &LuminosityFunctionHelper::set_precision)
    .def("computeDavisHuchraNormalization_magrange", &LuminosityFunctionHelper::computeDavisHuchraNormalization_magrange)
    .def("setLuminosityFunction_custom", &LuminosityFunctionHelper::setLuminosityFunction_custom)
    .def("setLuminosityFunction", &LuminosityFunctionHelper::setLuminosityFunction)
    .def("setCosmology", &LuminosityFunctionHelper::setCosmology)
    .def("setSunAbsoluteMagnitude", &LuminosityFunctionHelper::setSunAbsoluteMagnitude)
    .def("integral_nw_lumfun", &LuminosityFunctionHelper::integral_nw_lumfun)
    .def("integral_nw_lumfun_2m", &LuminosityFunctionHelper::integral_nw_lumfun_2m)
    .def("integral_lw_lumfun", &LuminosityFunctionHelper::integral_lw_lumfun)
    .def("integral_lw_lumfun_2m", &LuminosityFunctionHelper::integral_lw_lumfun_2m)
    .def("getWeight_1m", &LuminosityFunctionHelper::getWeight_1m)
    .def("getWeight_2m", &LuminosityFunctionHelper::getWeight_2m)
    .def("chooseWeight", &LuminosityFunctionHelper::chooseWeight)
    .def("computeAbsoluteMagnitudes", &LuminosityFunctionHelper::computeAbsoluteMagnitudes)
    .def("lumfun", &LuminosityFunctionHelper::lumfun)
    .def("updateLuminosityFunctionParameters", &LuminosityFunctionHelper::updateLuminosityFunctionParameters)
    .def("computeGalaxyWeights", &LuminosityFunctionHelper::computeGalaxyWeights)
    .def("computePhiStarNormalisation", &LuminosityFunctionHelper::computePhiStarNormalisation, (arg("maglimb"), arg("maglimf"), arg("cb"), arg("cf"), arg("dmin"), arg("dmax"), arg("J3")=40.0, arg("epsrel")=1e-3))
    ;

  class_<LuminosityDistance>("LuminosityDistance", init<Cosmo>())
    .def("dl", &LuminosityDistance::dl)
    ;
    
  class_<LuminosityFunctionLikelihood>("LuminosityFunctionLikelihood", 
				       init<const boost::python::object,const boost::python::object,const boost::python::object,
 				       const boost::python::object,const boost::python::object>()) 
    .def("computeLogLike", &LuminosityFunctionLikelihood::computeLogLike)
    .def("computeLogOne", &LuminosityFunctionLikelihood::computeLogOne)
    .def("findBestParameters", &LuminosityFunctionLikelihood::findBestParameters)
    .def("computeEffectiveNumber", &LuminosityFunctionLikelihood::computeEffectiveNumber)
    ;

  class_<Cosmo>("Cosmology")
    .add_property("omega_m", make_getter(&Cosmo::omega_m), make_setter(&Cosmo::omega_m))
    .add_property("omega_de", make_getter(&Cosmo::omega_de), make_setter(&Cosmo::omega_de))
    .add_property("w", make_getter(&Cosmo::w), make_setter(&Cosmo::w))
    ;

  enum_<WeighingScheme>("WeighingScheme")
    .value("USE_NUMBER_SCHEME", USE_NUMBER_SCHEME)
    .value("USE_LUMINOSITY_SCHEME", USE_LUMINOSITY_SCHEME)
    .value("USE_HOMOGENEOUS_LUMINOSITY_SCHEME", USE_HOMOGENEOUS_LUMINOSITY_SCHEME)
    .export_values()
    ;

  enum_<LuminosityFunctionHelper::LF_Type>("LF_Type")
    .value("LF_SCHECHTER_FUNCTION", LuminosityFunctionHelper::LF_SCHECHTER_FUNCTION)
    .value("LF_DOUBLE_POWER_LAW_FUNCTION", LuminosityFunctionHelper::LF_DOUBLE_POWER_LAW_FUNCTION)
    .value("LF_PYTHON_FUNCTION", LuminosityFunctionHelper::LF_PYTHON_FUNCTION)
    .value("LF_TRIPLE_POWER_LAW_FUNCTION", LuminosityFunctionHelper::LF_TRIPLE_POWER_LAW_FUNCTION)
    .value("LF_DOUBLE_SCHECHTER_FUNCTION", LuminosityFunctionHelper::LF_DOUBLE_SCHECHTER_FUNCTION)
    .export_values()
    ;
}
