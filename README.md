<!---
  EOgmaNeo
  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of EOgmaNeo is licensed to you under the terms described
  in the EOGMANEO_LICENSE.md file included in this distribution.
--->

# EOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)
## Introduction

Welcome to the EOgmaNeo library!

EOgmaNeo is Ogma Corp's embedded and event based version of [OgmaNeo](https://github.com/ogmacorp/OgmaNeo)

It is our _primary_ and _preferred_ implementation of Sparse Predictive Hierarchies, a fully online sequence predictor.

EOgmaNeo currently runs exclusive on the CPU, unlike OgmaNeo. However, for most tasks, it is much faster. It also performs better in terms of end-result.

EOgmaNeo performs some optimizations not yet present in OgmaNeo, resulting in a massive speed boost. For example, on weaker hardware such as the Raspberry Pi it will run happily at 60FPS with ~10,000,000 synapses.

We used this software to build a small self-contained online-learning self driving model car: [Blog post](https://ogma.ai/2017/06/self-driving-car-learns-online-and-on-board-on-raspberry-pi-3/)

The advantage of our software over standard Deep Learning techniques is primarily speed. A single Raspberry Pi 3 is enough to run simulations of networks with tens of millions of synapses at high framerates, while Deep Learning is often restricted to offline training on very large and expensive GPUs.

Bindings to Python are also included. The binding API approximately mimics the C++ API. Refer to README.md files in each subdirectory to discover more about each binding, and how to build and use them.

## Overview

**For a more detailed introduction, see [OVERVIEW.md](./OVERVIEW.md)**

EOgmaNeo is a fully online learning algorithm, so data must be passed in an appropriately streamed fashion.

The simplest usage of the predictive hierarchy involves calling:

```cpp
    // Compute system
    eogmaneo::ComputeSystem cs(4); // Number of threads to use, e.g. CPU Core count

    eogmaneo::Hierarchy h;

    // Layer descriptors
    std::vector<eogmaneo::LayerDesc> lds(6);

    // Layer size
    const int layerWidth = 4;
    const int layerHeight = 4;
    const int layerColumnSize = 32;

    for (int l = 0; l < lds.size(); l++) {
        lds[l]._width = layerWidth;
        lds[l]._height = layerHeight;
        lds[l]._columnSize = layerColumnSize;
        // ...
    }

    // Create the hierarchy
    h.create({ { 2, 2 } }, { 16 }, { true }, lds, 1234); // Input width x height, input column size, whether to predict, layer descriptors, and seed
```

You can then step the simulation with:

```cpp
    h.step(cs, sdrs, true); // Input SDRs, learning is enabled
```

And retrieve predictions with:

```cpp
    std::vector<int> predSDR = h.getPredictions(0); // Get SDR at first (0) visible layer index.
```

Important note: Inputs are presented in a _columnar SDR_ format. This format consists of a list of active units, each corresponding to a column of input.
This vector is in raveled form (1-D array of size width x height).

All data must be presented in this form. To help with conversion, we included a few "pre-encoders" - encoders that serve to transform various kinds of data into columnar SDR form.

Currently available pre-encoders:

- ImageEncoder
- KMeansEncoder
- GaborEncoder

You may need to develop your own pre-encoders depending on the task. Sometimes, data can be binned into columns without any real pre-encoding, such as bounded scalars.

## Requirements

EOgmaNeo requires: a C++1x compiler, and [CMake](https://cmake.org/).

The library has been tested extensively on:

- Windows using Microsoft Visual Studio 2013 and 2015,
- Linux using GCC 4.8 and upwards,
- Mac OSX using Clang, and
- Raspberry Pi 3, using Raspbian Jessie with GCC 4.8

### CMake

Version 3.1, and upwards, of [CMake](https://cmake.org/) is the required version to use when building the library.

## Building

The following commands can be used to build the EOgmaNeo library:

```bash
mkdir build; cd build
cmake -DBUILD_PREENCODERS=ON ..
make
```

The `cmake` command can be passed the following optional settings:

- `CMAKE_INSTALL_PREFIX` to determine where to install the library and header files. Default is a system-wide install location.
- `BUILD_PREENCODERS` to include the Random and Corner pre-encoders into the library.

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On Windows it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters.

## Examples

C++ examples can be found in the `source/examples` directory. Python, Java, and C# examples can be found in their sub-directories.

Refer to `README.md` files found in each sub-directory for further information.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/CONTRIBUTING.md) file for information on making contributions to EOgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [EOGMANEO_LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/EOGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/EOgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

EOgmaNeo Copyright (c) 2017-2018 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
