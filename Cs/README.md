<!---
  EOgmaNeo
  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of EOgmaNeo is licensed to you under the terms described
  in the EOGMANEO_LICENSE.md file included in this distribution.
--->

# C# bindings for EOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction

This [SWIG](http://www.swig.org/) binding provides an interface into the EOgmaNeo C++ library. Allowing C# code to gain access to the EOgmaNeo Sparse Predictive Hierarchies.

## Requirements

The same requirements that EOgmaNeo has, are required for this binding: a C++1x compiler and [CMake](https://cmake.org/).

Additionally this binding requires installation of a C# development environment (e.g. Mono or Visual Studio) and [SWIG](http://www.swig.org/) v3+

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example, ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (3+). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).
- Mac OSX can use Homebrew to install the latest SWIG (for example, see .travis/install_swig.sh Bash script).

#### OpenCV

As noted in the main README.md file, an [OpenCV](http://opencv.org/) interop class is built into the library if the CMake build process finds OpenCV on your system. 

**Note:** For the CMake build process to find OpenCV make sure that a `OpenCV_DIR` environment variable is set to the location of OpenCV, specifically the directory that contains the `OpenCVConfig.cmake` file. If this configuration file is not found the OpenCV interop code is not included in the library.

## Installation

Once the EOgmaNeo requirements have been setup. The following can be used to build the bindings library and supporting C# interface code:

> cd EOgmaNeo/Cs  
> mkdir build; cd build  
> cmake ..  
> make  

This will create a shared library (on Windows `EOgmaNeo.dll`, Linux and Mac OSX `libEOgmaNeo.so`) that provides the interface to the EOgmaNeo library. As well as supporting C# interface code that can be included in a C# project. This C# interface code _relatively_ imports the EOgmaNeo library.

## Importing and Setup

The EOgmaNeo module can be imported into C# code using:

```csharp
using eogmaneo;
```

## Contributions

Refer to the EOgmaNeo [CONTRIBUTING.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/CONTRIBUTING.md) file for details about contributing to EOgmaNeo, including the signing of the [Ogma Contributor Agreement](https://ogma.ai/wp-content/uploads/2016/09/OgmaContributorAgreement.pdf).

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [EOGMANEO_LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/EOGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

EOgmaNeo Copyright (c) 2017-2018 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
