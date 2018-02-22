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

set(EOGMANEO_MAJOR_VERSION 1)
set(EOGMANEO_MINOR_VERSION 2)
set(EOGMANEO_PATCH_VERSION 7)
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

option(BUILD_EXAMPLES OFF)
message(STATUS "Build examples: ${BUILD_EXAMPLES}")

option(BUILD_PREENCODERS OFF)
message(STATUS "Build pre-encoders: ${BUILD_PREENCODERS}")


############################################################################
# Find SFML
#set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2 COMPONENTS system network window graphics)

if(SFML_FOUND)
  #message(STATUS "SFML incs: " ${SFML_INCLUDE_DIR})
  #message(STATUS "SFML libs: " ${SFML_LIBRARIES})
  #message(STATUS "SFML deps: " ${SFML_DEPENDENCIES})
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

  file(GLOB_RECURSE EOGMANEO_POSITION2DENCODER_SRC "source/optional/Position2DEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_POSITION2DENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_SPARSEIMAGEENCODER_SRC "source/optional/SparseImageEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_SPARSEIMAGEENCODER_SRC})

  file(GLOB_RECURSE EOGMANEO_RLENCODER_SRC "source/optional/RLEncoder.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_RLENCODER_SRC})
endif()

if(SFML_FOUND)
  file(GLOB_RECURSE EOGMANEO_SFML_SRC "source/optional/VisAdapter.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_SFML_SRC})
endif()
if(OpenCV_FOUND)
  file(GLOB_RECURSE EOGMANEO_OPENCV_SRC "source/optional/OpenCVInterop.*")
  list(APPEND EOGMANEO_SRC ${EOGMANEO_OPENCV_SRC})
endif()

add_library(EOgmaNeo ${EOGMANEO_SRC})

if(SFML_FOUND)
  target_link_libraries(EOgmaNeo ${SFML_LIBRARIES})
endif()
if(OpenCV_FOUND)
  target_link_libraries(EOgmaNeo ${OpenCV_LIBS})
endif()

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

if(BUILD_EXAMPLES)
  add_executable(SaveLoad source/examples/SaveLoad.cpp)
  set_property(TARGET SaveLoad PROPERTY CXX_STANDARD 14)
  set_property(TARGET SaveLoad PROPERTY CXX_STANDARD_REQUIRED ON)
  target_link_libraries(SaveLoad EOgmaNeo)

  if(SFML_FOUND AND OpenCV_FOUND)
    add_executable(VideoPrediction source/examples/VideoPrediction.cpp)
    set_property(TARGET VideoPrediction PROPERTY CXX_STANDARD 14)
    set_property(TARGET VideoPrediction PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(VideoPrediction EOgmaNeo ${SFML_LIBRARIES} ${SFML_DEPENDENCIES} ${OpenCV_LIBS})
  endif()
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
