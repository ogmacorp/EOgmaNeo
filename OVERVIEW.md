<!---
  EOgmaNeo
  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of EOgmaNeo is licensed to you under the terms described
  in the EOGMANEO_LICENSE.md file included in this distribution.
--->

# EOgmaNeo Tutorial – Sine Wave Memorization

The following is a brief tutorial to walk you through the usage of the EOgmaNeo library. We will use EOgmaNeo to memorize a very simple time series – a sinosodial one. This example should in particular demonstrate how to present data to a EOgmaNeo Sparse Predictive Hierarchy (SPH).

In this tutorial, we will be using the EOgmaNeo Python bindings. The interface is nearly identical to the regular C++ library, and similar in other language bindings as well.

## Python bindings – Installation

Once you have downloaded the EOgmaNeo repository, you must simply proceed to the `/Python` subdirectory, and run Python on the setup.py script in that directory, e.g. `python3 setup.py install`. This should compile the library and install the Python binding. To test the installation, simply try importing `import eogmaneo`.

## Overview

EOgmaNeo provides an implementation of Sparse Predictive Hierarchies. These are online learning systems that are presented with the task of predicting data one timestep ahead of time. Given x(t), the hierarchy will return what it thinks x(t+1) will be.

An EOgmaNeo hierarchy consists of several layers. Each layer is 2-dimensional, it has a width, and a height. This is important for working with images, but for non-image tasks, one can simply ravel the data into a 2D setting. Connectivity patterns are also local – so you may need multiple layers to bridge between spatially distant information. The same goes for temporally distant information – more layers gives a larger memory horizon (exponentially increasing).

Each layer has an associated column size – this is the size of one column of neural activity. If width and height are the X and Y dimensions, then the column size is Z (although it has additional meaning). Within a column only one unit is active at a time. Each column is therefore a one-hot vector. A 2D grid of column is called a columnar SDR (sparse distributed representation). To represent a columnar SDR, EOgmaNeo uses a list of active unit indices – one index per column.

A columnar SDR is used by EOgmaNeo as a way of representing data. A particular columnar SDR state may represent object trajectories, motor commands, an image, a number of images in sequence (video), abstract concepts, timing information – any information can be mapped to a columnar SDR.

What is important about this particular format, however, is that it is both sparse, and locally sensitive. This means that very few units are active at a time, and similar columnar SDRs represent similar information.

These two properties permit both online learning and generalization, respectively. Online learning requires that the representation be sparse, as to avoid overlap in a representation. However, too sparse isn’t good either – it ends up acting like a lookup table. We find that a trade-off produces optimal results, resulting in both online learning capabilities as well as generalizability.

## Pre-Encoding

Now what remains is a problem of converting to a columnar SDR format. For this, we need a sort of “pre-encoder” (encoders are another concept used in EOgmaNeo, and are separate from pre-encoders for the most part). A pre-encoder maps from some data to a columnar SDR format.

We find that specific pre-encoders are good at specific tasks, although general-purpose pre-encoders exist as well. Sparse coding, in particular, is a good way to learn a particular encoder. However, sparse coding is often slow, so we instead found that simpler methods often work better in terms of processing requirements, while still delivering reasonable end results.

EOgmaNeo currently includes a small amount of pre-encoders. If you have any ideas for new pre-encoders, let us know!

- KMeansEncoder (random projection followed by inhibition)
- ImageEncoder (Encoder for image data)
- GaborEncoder (Alternative encoder for images with fixed Gabor filters)

Once a pre-encoder maps the data to a columnar SDR, the data can be learned from and predicted by a EOgmaNeo hierarchy. Sometimes we also need a reverse mapping for the pre-encoder (pre-decoder), in order to retrieve results. KMeansEncoder, for instance, is reversible, using its ```reconstruct(...)``` function.