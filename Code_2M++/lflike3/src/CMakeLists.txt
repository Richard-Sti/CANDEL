SET(lflike_SRCS _lflike.cpp likelihood.cpp lfhelper.cpp gamminc.f90 lfimpl.cpp lf_schechter.cpp lf_dp_schechter.cpp lf_dp_law.cpp lf_tp_law.cpp interpolate.cpp)

add_library(_lflike MODULE ${lflike_SRCS})
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_lflike ${Boost_LIBRARIES} ${PYTHON_LIBRARY} ${GSL_LIBRARIES})

# This is a Python module
SET_TARGET_PROPERTIES(_lflike PROPERTIES PREFIX "")

if (WIN32 AND NOT CYGWIN)
  SET_TARGET_PROPERTIES(_lflike PROPERTIES SUFFIX ".pyd")
endif (WIN32 AND NOT CYGWIN)

INSTALL(TARGETS _lflike
  LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/lflike3
)

INSTALL(FILES python/__init__.py
  DESTINATION ${PYTHON_SITE_PACKAGES}/lflike3
)
