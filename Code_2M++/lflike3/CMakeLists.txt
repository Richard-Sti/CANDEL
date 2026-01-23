cmake_minimum_required(VERSION 2.8)

project(lflike3)

enable_language(Fortran)

include(InstallRequiredSystemLibraries)
INCLUDE (CheckIncludeFile)
include(CheckIncludeFileCXX)
INCLUDE (CheckIncludeFiles)
INCLUDE(CheckLibraryExists)
include(f77_mangling.cmake)

FIND_PACKAGE(Boost 1.46.0 REQUIRED COMPONENTS python)
FIND_PACKAGE(PythonLibs 2.6 REQUIRED)
FIND_PACKAGE(PythonInterp 2.6 REQUIRED)

find_library(GSL_LIBRARY gsl)
find_library(GSLCBLAS_LIBRARY gslcblas)

set(GSL_LIBRARIES ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})


# Discover where to put packages
execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/f77_mangling.hpp 
  "${DEFINE_F77_FUNC}\n"
  "${DEFINE_F77_FUNC_}\n")

add_subdirectory(src)
