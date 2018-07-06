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

## Sine Wave

Now that we know about pre-encoding, we can tackle the simple task of memorizing a sine wave. For this task, we want to proceed one timestep at a time (streaming), such that the sine wave is presented as a sequence of scalars. We therefore need a way to map a scalar to a chunked SDR. There are two options that seem appropriate for this task: Use a RandomEncoder to map a single scalar to a chunked SDR, or perform a “raw” encoding, by simply bucketing the scalar (which is bounded in ```[-1, 1]``` since it is a sine wave) into a single chunk. In this situation, a random encoder basically just buckets the scalar at random intervals, while with a manual approach we can set the bucket interval precisely, so we will choose this one for now.

With the bucket approach, encoding a bounded scalar is as simple as rescaling the value such that it fits into buckets uniformly. A single chunk can represent this bucketed scalar, resulting in a chunk where the position of the active unit linearly encodes the scalar.

Since EOgmaNeo takes arrays of chunks as input (a chunked SDR), we can create the required encoding by the following Python code:

```python
chunkSize = 8

valueToEncode = 0.3 # Test value

unitsPerChunk = chunkSize * chunkSize

bounds = (-1.0, 1.0) # Range of value

# Single-chunk SDR
chunkedSDR = [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (unitsPerChunk - 1) + 0.5) ]
```

However, before we use this, we first need to build a hierarchy. To do this, we need to define several LayerDesc structures. These describe each layer, with several tuneable parameters. The purpose of each parameter is beyond the scope of this tutorial. Defaults will suffice for most tasks, but here is an example:

```python
lds = []

for i in range(3):
    ld = eogmaneo.LayerDesc()

    ld._width = 16
    ld._height = 16
    ld._chunkSize = 8
    ld._forwardRadius = ld._backwardRadius = 12
    ld._alpha = 0.1
    ld._beta = 0.04
    ld._delta = 0.0
    ld._temporalHorizon = 2

    lds.append(ld)
```

Then, to use these 3 layers to create a 3-layer hierarchy, we do the following:

```python
h = eogmaneo.Hierarchy()

h.create([ ( chunkSize, chunkSize ) ], [ chunkSize ], [ True ], lds, 123)
```

This first parameter is simply a list of input dimension tuples. Note that this is in number of units, not chunks. So, to feed our single chunk, we need a chunkSize by chunkSize input layer.

The second parameter is the chunk size of each input layer. Naturally, our single chunk is of size chunkSize.

The third parameter specifies which input layer to predict. This is mostly for optimization purposes, some layers don’t need to be predicted (input only), and can therefore be ignored.

The fourth parameter is simply the list of `LayerDesc`.

Finally, the fifth parameter is a seed for the internal random number generator.

We will also need a `ComputeSystem` object, which contains a thread pool. Simply create one with your desired thread count (typically the number of cores your machine has):

```python
system = eogmaneo.ComputeSystem(4)
```

Now that we have a hierarchy, we can present it with our sine wave sequence using the encoding method shown previously:

```python
# Present the sine wave sequence for 1000 steps
for t in range(1000):
    valueToEncode = np.sin(t * 0.04 * 2.0 * np.pi) # Test value

    # Single-chunk SDR
    chunkedSDR = [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (unitsPerChunk - 1) + 0.5) ]

    h.step([ chunkedSDR ], system, True)
```

The step function runs a single timestep of the hierarchy, and automatically generates the predictions of the next timestep. It takes a list of chunked SDRs (one for each input layer), the system object, and a Boolean that determines whether or not learning is enabled.

To retrieve a predicted SDR, we simply call:

```python
predSDR = h.getPrediction(0) # First (only in this case) input layer prediction
```

We can then recall the information by looping the hierarchy on its own predictions, with learning disabled:

```python
# Recall
for t in range(100):
    predSDR = h.getPrediction(0) # First (only in this case) input layer prediction

    # Decode value
    value = predSDR[0] / float(unitsPerChunk - 1) * (bounds[1] - bounds[0]) + bounds[0]

    print(value)

    h.step([ predSDR ], system, False)
```

This will print a list of sine wave values.

Finally, we can save the result like so:

```python
h.save("sineSave.eohr")
```

To load it again, we simply call ```h.load(...)``` instead of ```h.create(...)``` . 

If you would like to gain a better understanding of how the hierarchy works, as well as peer inside of whatever network you are working with, try [NeoVis](https://github.com/ogmacorp/NeoVis), the EOgmaNeo visualizer!

This concludes the basic tutorial. We would love to hear about any projects you make with this!

The complete program code is available in the Python directory of the repository (sineWaveExample.py).