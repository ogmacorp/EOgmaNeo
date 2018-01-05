<!---
  EOgmaNeo
  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.

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

Bindings to Python, C#, and Java are also included. The binding APIs approximately mimic the C++ API. Refer to README.md files in each subdirectory to discover more about each binding, and how to build and use them.

## Overview

**For a more detailed introduction, see [TUTORIAL.md](./TUTORIAL.md)**

EOgmaNeo is a fully online learning algorithm, so data must be passed in an appropriately streamed fashion.

The simplest usage of the predictive hierarchy involves calling:

```cpp
    // Compute system
    eogmaneo::ComputeSystem system(4); // Number of threads to use, e.g. CPU Core count

    eogmaneo::Hierarchy h;

    // Layer descriptors
    std::vector<eogmaneo::LayerDesc> lds(6);

    // Layer size
    const int layerWidth = 32;
    const int layerHeight = 32;
    const int layerChunkSize = 8;

    for (int l = 0; l < lds.size(); l++) {
        lds[l]._width = layerWidth;
        lds[l]._height = layerHeight;
        lds[l]._chunkSize = layerChunkSize;
        // ...
    }

    // Create the hierarchy
    h.create({ { 16, 16 } }, { 16 }, { true }, lds, 1234);
```

You can then step the simulation with:

```cpp
    h.step(sdrs, system, true); // Input SDRs, learning is enabled
```

And retrieve predictions with:

```cpp
    std::vector<int> predsdr = h.getPrediction(0); // Get SDR at first (0) visible layer index.
```

Important note: Inputs are presented in a _chunked SDR_ format. This format consists of a list of active units, each corresponding to a chunk (or _tile_) of input.
This vector is in raveled form (1-D array of size width x height).

Here is an image to help describe the input format: [Chunked SDR](./chunkedSDR.png)

All data must be presented in this form. To help with conversion, we included a few "pre-encoders" - encoders that serve to transform various kinds of data into chunked SDR form.

Currently available pre-encoders:

- ImageEncoder
- KMeansEncoder
- LineSegmentEncoder (OpenCV)
- FastFeatureDetector (OpenCV)

You may need to develop your own pre-encoders depending on the task. Sometimes, data can be binned into chunks without any real pre-encoding, such as bounded scalars.

Further examples of using the pre-encoders and EOgmaNeo can be found in the `source/examples` directory. A `README.md` file in that directory explains each example application found there.

### Optional features

#### SFML (NeoVis)

The EOgmaNeo library contains the client-side code required by the [NeoVis](https://github.com/ogmacorp/NeoVis) hierarchy visualization tool. 

The CMakeLists.txt file tries to find [SFML](https://www.sfml-dev.org/) headers and libraries on your system, to link to and use SFML's network interface functionality. If it can be found the VisAdapter code is added into the library, otherwise it is left out.

The NeoVis client-side VisAdapter class can be used with EOgmaNeo and within an application to enable real-time streaming of hierarchy information to the NeoVis application.

As mentioned in the NeoVis [Readme.md](https://github.com/ogmacorp/NeoVis/blob/master/README.md) the VisAdapter is constructed and linked to an EOgmaNeo hierarchy using the following C++ example (language bindings contain similar interfaces):

```cpp
    // Construct and setup an EOgmaNeo hierarchy
    eogmaneo.Hierarchy hierarchy;
    ...

    // Create a NeoVis adapter and link to the hierarchy
    eogmaneo.VisAdapter neoVisAdapter;
    neoVisAdapter.create(hierarchy, 54000); // Using default socket port
```

Sending the current EOgmaNeo hierarchy state through the adapter to the NeoVis application can be simple done using the following example:

```cpp
    // Perform the usual hierarchy stepping
    hierarchy.step(...);

    // Send the current state of the hierarchy through
    // the adapter to the NeoVis application
    neoVisAdapter.update();
```

#### OpenCV Interop

An [OpenCV](http://opencv.org/) interop class is built into the library if the CMake build process finds OpenCV on your system.

**Note:** For the CMake build process to find OpenCV make sure that a `OpenCV_DIR` environment variable is set to the location of OpenCV, specifically the directory that contains the `OpenCVConfig.cmake` file. If this configuration file is not found the OpenCV interop code is not included in the library.

Currently it contains a handful of calls into the following OpenCV C++ functions:

- [CannyEdgeDetection](http://docs.opencv.org/3.2.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de)
- [Threshold](http://docs.opencv.org/3.2.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)
- [AdaptiveThreshold](http://docs.opencv.org/3.2.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
- [GaborFilter](http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599) (cv::filter2D)
- [LineSegmentDetector](http://docs.opencv.org/3.2.0/dd/d1a/group__imgproc__feature.html#ga6b2ad2353c337c42551b521a73eeae7d) SDR pre-encoder
- [FastFeatureDetector](http://docs.opencv.org/3.2.0/d5/d51/group__features2d__main.html#gaf3185c9bd7496ba7a629add75fb371ad) SDR pre-encoder

Be aware that all these functions contain certain remapping of input arrays into OpenCV Mat types, with appropriate remapping when results are output. Refer to the OpenCVInterop.cpp file to see what input and output mappings occur, and what value ranges are expected in the input array(s).

**Notes:**
- The `LineSegmentDetector` contains extra functionality that takes detected lines and forms them into a sparse chunked representation as it's output. Therefore, the LineSegmentDetector acts as a pre-encoder for an EOgmaNeo hierarchy.
- Similar to the `LineSegmentDetector`, the `FastFeatureDetector` contains extra functionality that takes detected corner key points and forms them into a sparse chunked representation as its output. Therefore, it acts as a pre-encoder for an EOgmaNeo hierarchy.
- On Mac OSX, Homebrew can be used to cleanly install OpenCV, for example `brew install opencv3 --with-contrib --with-python3 --with-ffmpeg`


## Requirements

EOgmaNeo requires: a C++1x compiler, and [CMake](https://cmake.org/).

Optional requirements include [OpenCV](http://opencv.org/) (for additional pre-encoders) and [SFML](https://www.sfml-dev.org/) (for connecting to the [NeoVis](https://github.com/ogmacorp/NeoVis) visualization tool).

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
- `BUILD_EXAMPLES` to include the C++ examples found in the `source/examples/` directory.

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

EOgmaNeo Copyright (c) 2017 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
