#ifndef _LFLIKE_SHARED_SYMBOLS_DEF_HPP
#define _LFLIKE_SHARED_SYMBOLS_DEF_HPP

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
  #define LFLIKE_DLL_IMPORT __declspec(dllimport)
  #define LFLIKE_DLL_EXPORT __declspec(dllexport)
  #define LFLIKE_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define LFLIKE_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define LFLIKE_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define LFLIKE_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define LFLIKE_DLL_IMPORT
    #define LFLIKE_DLL_EXPORT
    #define LFLIKE_DLL_LOCAL
  #endif
#endif

#endif
