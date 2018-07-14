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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/")

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

if(PYTHON_VERSION)
    message("Python target version: ${PYTHON_VERSION}")            
endif()

option(BUILD_PREENCODERS ON)
message(STATUS "Build pre-encoders: ${BUILD_PREENCODERS}")


############################################################################
# Add the EOgmaNeo library

set(EOGMANEO_INCLUDE_DIR "../source/eogmaneo;../source/optional")

include_directories(${EOGMANEO_INCLUDE_DIR})

file(GLOB_RECURSE EOGMANEO_SRC
    "../source/eogmaneo/*.h"
    "../source/eogmaneo/*.cpp"
)

if (BUILD_PREENCODERS)
  file(GLOB_RECURSE EOGMANEO_IMAGEENCODER_SRC "../source/optional/ImageEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_IMAGEENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_KMEANSENCODER_SRC "../source/optional/KMeansEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_KMEANSENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_GABORENCODER_SRC "../source/optional/GaborEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_GABORENCODER_SRC})
endif()

add_library(EOgmaNeo ${EOGMANEO_SRC})

############################################################################
# Find SWIG and setup building the Python bindings to EOgmaNeo library

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()


############################################################################
# Find Python package

# Are we dealing with Anaconda?
set(CONDA_COMMAND conda info --root)
execute_process(
    COMMAND ${CONDA_COMMAND}
    RESULT_VARIABLE _CONDA_RESULT
    OUTPUT_VARIABLE _CONDA_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(${_CONDA_RESULT} STREQUAL "0")
    string(TOLOWER ${_CONDA_OUTPUT} _CONDA_OUTPUT)
    string(REPLACE "\\" "/" _CONDA_OUTPUT ${_CONDA_OUTPUT})
    string(FIND "${_CONDA_OUTPUT}" "conda" ANACONDA_PYTHON_FOUND)
endif()

if(ANACONDA_PYTHON_FOUND)
    set(ANACONDA_PYTHON_DIR ${_CONDA_OUTPUT})
    message(STATUS "Found anaconda in ${ANACONDA_PYTHON_DIR}")

    set(_PYTHON_COMMAND python --version)
    execute_process(
        COMMAND ${_PYTHON_COMMAND}
        ERROR_VARIABLE _PYTHON_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "Python ([0-9]+)[.]([0-9]+)[.]([0-9]+)" _PYTHON_VERSION_FOUND "${_PYTHON_ERROR}")
    set(_PYTHON_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(_PYTHON_VERSION_MINOR ${CMAKE_MATCH_3})
    set(_PYTHON_VERSION_PATCH ${CMAKE_MATCH_0})

    if(WIN32)
        set(ANACONDA_PYTHON_VERSION ${_PYTHON_VERSION_MAJOR}${_PYTHON_VERSION_MINOR})
        set(_PYTHON_LIB "libs/python${ANACONDA_PYTHON_VERSION}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(ANACONDA_PYTHON_VERSION ${_PYTHON_VERSION_MAJOR}.${_PYTHON_VERSION_MINOR})
        if(${_PYTHON_VERSION_MAJOR} MATCHES 2)
            set(_PYTHON_EXTENSION "")
        else()
            set(_PYTHON_EXTENSION "m")
        endif()
        set(_PYTHON_ID "python${ANACONDA_PYTHON_VERSION}${_PYTHON_EXTENSION}")
        set(_PYTHON_LIB "lib/${CMAKE_SHARED_LIBRARY_PREFIX}python${ANACONDA_PYTHON_VERSION}${_PYTHON_EXTENSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    if(NOT DEFINED ENV{CONDA_DEFAULT_ENV} OR $ENV{CONDA_DEFAULT_ENV} STREQUAL "base")
        set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/include/${_PYTHON_ID}" CACHE INTERNAL "")
        set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/${_PYTHON_LIB}" CACHE INTERNAL "")
    else()
        set(_CONDA_DEFAULT_ENV $ENV{CONDA_DEFAULT_ENV})
        string(REPLACE "\\" "/" _CONDA_DEFAULT_ENV ${_CONDA_DEFAULT_ENV})
        set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/${_CONDA_DEFAULT_ENV}/include/${_PYTHON_ID}" CACHE INTERNAL "")
        set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/${_CONDA_DEFAULT_ENV}/${_PYTHON_LIB}" CACHE INTERNAL "")
    endif()

    message("PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
    message("PYTHON_LIBRARY = ${PYTHON_LIBRARY}")

    set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
    set(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")
    set(FOUND_PYTHONLIBS TRUE)
else()
    if(DEFINED ENV{TRAVIS} AND DEFINED ENV{TRAVIS_PYTHON_VERSION})
        message(STATUS "Finding PythonLibs $ENV{TRAVIS_PYTHON_VERSION}")
        if("$ENV{TRAVIS_PYTHON_VERSION}" STREQUAL "2.7")
            find_package(PythonLibs $ENV{TRAVIS_PYTHON_VERSION})
        else()
            if(NOT PYTHON_VERSION)
                find_package(PythonLibs)
            else()
                find_package(PythonLibs ${PYTHON_VERSION})
            endif()
        endif()
    else()
        if(NOT PYTHON_VERSION)
            find_package(PythonLibs)
        else()
            find_package(PythonLibs ${PYTHON_VERSION})
        endif()
    endif()
endif()

include_directories(${PYTHON_INCLUDE_DIR})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}")
set(CMAKE_SWIG_FLAGS "-I\"${PYTHON_INCLUDE_DIR}\" -I\"${EOGMANEO_INCLUDE_DIR}\"")

if(BUILD_PREENCODERS)
  list(APPEND CMAKE_SWIG_FLAGS -DBUILD_PREENCODERS)
  add_definitions(-DBUILD_PREENCODERS)
endif()
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set_source_files_properties(pyeogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_module(eogmaneo python pyeogmaneo.i)
swig_link_libraries(eogmaneo ${PYTHON_LIBRARIES} EOgmaNeo)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SWIG_OUTDIR})
install(TARGETS _eogmaneo DESTINATION ${CMAKE_SWIG_OUTDIR})
install(FILES ${CMAKE_BINARY_DIR}/eogmaneo.py DESTINATION ${CMAKE_SWIG_OUTDIR})
