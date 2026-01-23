#ifndef __COSMOTOOL_CONFIG_HPP
#define __COSMOTOOL_CONFIG_HPP

#include <string>
#include <stdint.h>
#include <exception>
#include <cstring>


#define NUMDIMS 3
#define NUMCUBES 8

  /**
   * Base type to specity at what precision we
   * must achieve computations.
   */
  typedef double ComputePrecision;
  /**
   * Coordinate type (should be a 3-array).
   */
  typedef double Coordinates[NUMDIMS];

  /* 
   * Single precision coordinates.
   */
  typedef float FCoordinates[NUMDIMS];
  
  /**
   * This function is used whenever one needs a general
   * conversion between mass and luminosity (or the opposite).
   * It should take a "mass" (or luminosity) in input, a unit is
   * given to convert this mass into solar units. The output should
   * be the "luminosity" (or mass), in solar units.
   */
  typedef double (*BiasFunction)(double mass, double unit);
  
  /**
   * Function to copy the coordinates "a" into "b".
   */
  inline void copyCoordinates(const Coordinates& a, Coordinates& b)
  {
    memcpy(b, a, sizeof(a));
  }

  /**
   * Base exception class for all exceptions handled by
   * this library.
   */
  class Exception : public std::exception
  {
  public:
    Exception(const std::string& mess)
      : msg(mess), msgok(true) {}
    Exception()
      : msgok(false) {}

    virtual ~Exception() throw () {}

    const char *getMessage() const { return msgok ? msg.c_str() : "No message"; };
    virtual const char *what() const throw () { return msgok ? msg.c_str() : "What 'what' ?"; };

  private:
    std::string msg;
    bool msgok;
  };

  /**
   * Exception raised when an invalid argument has been
   * passed to a function of the library.
   */
  class InvalidArgumentException : public Exception
  {
  public:
    InvalidArgumentException(const std::string& mess)
      : Exception(mess) {}
    InvalidArgumentException()
      : Exception() {}
  };

  /**
   */
  class InvalidRangeException : public Exception
  {
  public:
    InvalidRangeException(const std::string& mess)
      : Exception(mess) {}
    InvalidRangeException()
      : Exception() {}
  };
  
  /**
   */
  class NoSuchFileException : public Exception
  {
  public:
    NoSuchFileException(const std::string& mess)
      : Exception(mess) {}
    NoSuchFileException()
      : Exception() {}
  };

  /**
   */
  class InvalidFileFormatException : public Exception
  {
  public:
    InvalidFileFormatException(const std::string& mess)
      : Exception(mess) {}
    InvalidFileFormatException()
      : Exception() {}
  };

  class EndOfFileException: public Exception
  {
  public:
    EndOfFileException(const std::string& mess)
      : Exception(mess) {}
    EndOfFileException()
      : Exception() {}
  };

  class FilesystemFullException: public Exception
  {
  public:
    FilesystemFullException(const std::string& mess)
      : Exception(mess) {}
    FilesystemFullException()
      : Exception() {}
  };

#endif
