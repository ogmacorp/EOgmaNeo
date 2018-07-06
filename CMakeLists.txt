# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(EOgmaNeo)
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(EOGMANEO_MAJOR_VERSION 1)
set(EOGMANEO_MINOR_VERSION 3)
set(EOGMANEO_PATCH_VERSION 0)
set(EOGMANEO_VERSION ${EOGMANEO_MAJOR_VERSION}.${EOGMANEO_MINOR_VERSION}.${EOGMANEO_PATCH_VERSION})
 
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()
message(STATUS "Bitness: ${BITNESS}")

if(NOT CMAKE_BUILD_TYPE)
  message("CMAKE_BUILD_TYPE not set, setting it to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(BUILD_PREENCODERS ON)
message(STATUS "Build pre-encoders: ${BUILD_PREENCODERS}")


############################################################################
# Add the EOgmaNeo library

set(EOGMANEO_INCLUDE_DIR "source/eogmaneo;source/optional")

include_directories(${EOGMANEO_INCLUDE_DIR})

file(GLOB_RECURSE EOGMANEO_SRC
    "source/eogmaneo/*.h"
    "source/eogmaneo/*.cpp"
)

if (BUILD_PREENCODERS)
  file(GLOB_RECURSE EOGMANEO_IMAGEENCODER_SRC "source/optional/ImageEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_IMAGEENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_KMEANSENCODER_SRC "source/optional/KMeansEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_KMEANSENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_GABORENCODER_SRC "source/optional/GaborEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_GABORENCODER_SRC})
endif()

add_library(EOgmaNeo ${EOGMANEO_SRC})

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set_property(TARGET EOgmaNeo PROPERTY CXX_STANDARD 14)
set_property(TARGET EOgmaNeo PROPERTY CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  if(BITNESS EQUAL 64)
    set_target_properties(EOgmaNeo PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()
    
# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include/oegmaneo CACHE PATH "Installation directory for header files")

message(STATUS "CMake install prefix = ${CMAKE_INSTALL_PREFIX}")

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/EOgmaNeo)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
 
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()
 
# Add all targets to the build-tree export set
export(TARGETS EOgmaNeo
  FILE "${PROJECT_BINARY_DIR}/EOgmaNeoTargets.cmake")
 
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE EOgmaNeo)
 
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(EOgmaNeoConfig.cmake.in
  "${PROJECT_BINARY_DIR}/EOgmaNeoConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS "\${EOGMANEO_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(EOgmaNeoConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EOgmaNeoConfig.cmake" @ONLY)

configure_file(EOgmaNeoConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/EOgmaNeoConfigVersion.cmake" @ONLY)
 
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EOgmaNeoConfig.cmake"
  "${PROJECT_BINARY_DIR}/EOgmaNeoConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")

# Library install target
install(TARGETS EOgmaNeo
        EXPORT EOgmaNeoTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Headers, incl. schema headers, install targets
install(DIRECTORY "${PROJECT_SOURCE_DIR}/source/"
        EXPORT EOgmaNeoTargets
        DESTINATION include/eogmaneo
        FILES_MATCHING PATTERN "*.h*")

# 3rd party headers install targets
install(DIRECTORY "${3RDPARTY_PATH}/include/"
        EXPORT EOgmaNeoTargets
        DESTINATION include/eogmaneo
        FILES_MATCHING PATTERN "*.h*")
 
install(EXPORT EOgmaNeoTargets
        DESTINATION "${INSTALL_CMAKE_DIR}")

# Uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
