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
#message("Bitness: ${BITNESS}")


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
# Find Java and JNI

find_package(Java)
find_package(JNI)

include(UseJava)

if (Java_FOUND)
    message(STATUS "Found Java version ${Java_VERSION}")
endif()

if (JNI_FOUND)
    message(STATUS "Found Java JNI in ${JNI_INCLUDE_DIRS}")
    message(STATUS "Found Java JNI lib ${JNI_LIBRARIES}")
endif()

include_directories(${JNI_INCLUDE_DIRS})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}/src/main/java/com/ogmacorp/eogmaneo")
set(CMAKE_SWIG_FLAGS -package com.ogmacorp.eogmaneo -I\"${EOGMANEO_INCLUDE_DIR}\")

if(SFML_FOUND)
  list(APPEND CMAKE_SWIG_FLAGS -DSFML_FOUND)
  add_definitions(-DSFML_FOUND)
endif()
if (OpenCV_FOUND)
  list(APPEND CMAKE_SWIG_FLAGS -DOPENCV_FOUND)
  add_definitions(-DOPENCV_FOUND)
endif()
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set_source_files_properties(jeogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_module(JEOgmaNeo java jeogmaneo.i)
swig_link_libraries(JEOgmaNeo ${JNI_LIBRARIES} EOgmaNeo)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/classes")
add_custom_command(
    TARGET JEOgmaNeo POST_BUILD
    COMMAND cmake -E echo "Compiling Java files"
    COMMAND ${Java_JAVAC_EXECUTABLE} -d classes src/main/java/com/ogmacorp/eogmaneo/*.java
    COMMAND cmake -E echo "Creating Jar file"
    COMMAND ${Java_JAR_EXECUTABLE} cfM EOgmaNeo.jar -C classes .
    COMMAND cmake -E echo "Copying EOgmaNeo library"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:EOgmaNeo> "${CMAKE_BINARY_DIR}"
)