# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
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

option(BUILD_PREENCODERS OFF)
message(STATUS "Build pre-encoders: ${BUILD_PREENCODERS}")


############################################################################
# Add SFML
find_package(SFML 2 COMPONENTS system network)

if(SFML_FOUND)
  #message(STATUS "SFML incs: " ${SFML_INCLUDE_DIR})
  #message(STATUS "SFML libs: " ${SFML_LIBRARIES})
  include_directories(${SFML_INCLUDE_DIR})
endif()


############################################################################
# Find OpenCV

find_package(OpenCV HINTS /usr/local/opt/opencv3)

if(OpenCV_FOUND)
    #message(STATUS "Found existing OpenCV in ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


############################################################################
# Add the EOgmaNeo library

set(EOGMANEO_INCLUDE_DIR "../source/eogmaneo;../source/optional")

include_directories(${EOGMANEO_INCLUDE_DIR})

file(GLOB_RECURSE EOGMANEO_SRC
    "../source/eogmaneo/*.h"
    "../source/eogmaneo/*.cpp"
)

if(BUILD_PREENCODERS)
  file(GLOB_RECURSE EOGMANEO_IMAGEENCODER_SRC "../source/optional/ImageEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_IMAGEENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_KMEANSENCODER_SRC "../source/optional/KMeansEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_KMEANSENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_POSITION2DENCODER_SRC "../source/optional/Position2DEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_POSITION2DENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_SPARSEIMAGEENCODER_SRC "../source/optional/SparseImageEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_SPARSEIMAGEENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_RLENCODER_SRC "../source/optional/RLEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_RLENCODER_SRC})
endif()

if(SFML_FOUND)
  file(GLOB_RECURSE EOGMANEO_SFML_SRC "../source/optional/VisAdapter.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_SFML_SRC})
endif()
if(OpenCV_FOUND)
  file(GLOB_RECURSE EOGMANEO_OPENCV_SRC "../source/optional/OpenCVInterop.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_OPENCV_SRC})
endif()

add_library(EOgmaNeo ${EOGMANEO_SRC})

if(SFML_FOUND)
  target_link_libraries(EOgmaNeo ${SFML_LIBRARIES})
endif()
if(OpenCV_FOUND)
    target_link_libraries(EOgmaNeo ${OpenCV_LIBS})
endif()

############################################################################
# Find SWIG and setup building the Python bindings to OgmaNeo library

find_package(SWIG REQUIRED)
include(UseSWIG)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_SOURCE_DIR}/src/com/ogmacorp/eogmaneo")
set(CMAKE_SWIG_FLAGS -namespace eogmaneo -outdir ${CMAKE_SWIG_OUTDIR} -I\"${EOGMANEO_INCLUDE_DIR}\")

if(BUILD_PREENCODERS)
  list(APPEND CMAKE_SWIG_FLAGS -DBUILD_PREENCODERS)
  add_definitions(-DBUILD_PREENCODERS)
endif()
if(SFML_FOUND)
  list(APPEND CMAKE_SWIG_FLAGS -DSFML_FOUND)
  add_definitions(-DSFML_FOUND)
endif()
if(OpenCV_FOUND)
  list(APPEND CMAKE_SWIG_FLAGS -DOPENCV_FOUND)
  add_definitions(-DOPENCV_FOUND)
endif()
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
if(BUILD_SHARED_LIBS)
    add_definitions(-DOGMA_DLL)
endif()

set_source_files_properties(cseogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_module(CsEOgmaNeo csharp cseogmaneo.i)
swig_link_libraries(CsEOgmaNeo EOgmaNeo)

set_target_properties(CsEOgmaNeo PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SWIG_OUTDIR})

if(BUILD_BUNDLE)
    set_target_properties(CsEOgmaNeo PROPERTIES BUNDLE TRUE)
    set_target_properties(CsEOgmaNeo PROPERTIES SUFFIX “.bundle”)
endif()
