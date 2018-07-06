<!---
  EOgmaNeo
  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of EOgmaNeo is licensed to you under the terms described
  in the EOGMANEO_LICENSE.md file included in this distribution.
--->

# Python bindings for EOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction

This [SWIG](http://www.swig.org/) binding provides an interface into the EOgmaNeo C++ library. Allowing Python scripts to gain access to the EOgmaNeo Sparse Predictive Hierarchies.

## Requirements

The same requirements that EOgmaNeo has, are required for the bindings: a C++1x compiler and [CMake](https://cmake.org/).

Additionally this binding requires an installation of [SWIG](http://www.swig.org/) v3+

These bindings have been tested using:

| Distribution | Operating System (Compiler) |
| --- | ---:|
| Python 2.7 | Linux (GCC 4.8+) |
| Python 2.7 | Mac OSX |
| Anaconda Python 2.7 3.4 & 3.5 | Linux (GCC 4.8+) |
| Anaconda Python 3.5 | Windows (MSVC 2015) |

Further information on Python compatible Windows compilers can be found [here](https://wiki.python.org/moin/WindowsCompilers).

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (v3). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).
- Mac OSX can use Homebrew to install the latest SWIG (for example, see .travis/install_swig.sh Bash script).

## Installation

The following example can be used to build the Python package:

> cd EOgmaNeo/Python  
> python3 setup.py install --user  

or create a wheel file for installation via pip:

> cd EOgmaNeo/Python  
> python3 setup.py bdist_wheel  
> pip3 install dist/*.whl --user  

The `setup.cfg` file defines additional CMake build variables. Currently they enable the building of pre-encoders, and disables building of the [NeoVis](https://github.com/ogmacorp/NeoVis) link.

## Importing and Setup

The EOgmaNeo Python module can be imported using:

```python
import eogmaneo
```

Refer to the `sineWaveExample.py` example for further details.

## Contributions

Refer to the EOgmaNeo [CONTRIBUTING.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/CONTRIBUTING.md) file for details about contributing to EOgmaNeo, including the signing of the [Ogma Contributor Agreement](https://ogma.ai/wp-content/uploads/2016/09/OgmaContributorAgreement.pdf).

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [EOGMANEO_LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/EOGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

EOgmaNeo Copyright (c) 2017-2018 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
