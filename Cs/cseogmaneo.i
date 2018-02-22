// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

%begin %{
#include <cmath>
#include <iostream>
%}
%module eogmaneo

%include "std_pair.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "ComputeSystem.h"
#include "Layer.h"
#include "Hierarchy.h"
#include "Preprocessing.h"
#ifdef BUILD_PREENCODERS
#include "KMeansEncoder.h"
#include "ImageEncoder.h"
#include "Position2DEncoder.h"
#include "SparseImageEncoder.h"
#include "RLEncoder.h"
#endif
#ifdef SFML_FOUND
#include "VisAdapter.h"
#endif
#ifdef OPENCV_FOUND
#include "OpenCVInterop.h"
#endif
%}

%template(StdVeci) std::vector<int>;
%template(Std2DVeci) std::vector<std::vector<int> >;
%template(StdPairi) std::pair<int, int>;
%template(StdPairf) std::pair<float, float>;
%template(StdVecPairi) std::vector<std::pair<int, int> >;
%template(StdVecLayerDesc) std::vector<eogmaneo::LayerDesc>;
%template(StdVecf) std::vector<float>;
%template(Std2DVecf) std::vector<std::vector<float> >;
%template(StdVecb) std::vector<bool>;

%ignore eogmaneo::ForwardWorkItem;
%ignore eogmaneo::BackwardWorkItem;

%include "ComputeSystem.h"
%include "Layer.h"
%include "Hierarchy.h"
%include "Preprocessing.h"
#ifdef BUILD_PREENCODERS
%include "KMeansEncoder.h"
%include "ImageEncoder.h"
%include "Position2DEncoder.h"
%include "SparseImageEncoder.h"
%include "RLEncoder.h"
#endif

#ifdef SFML_FOUND
%include "VisAdapter.h"
#endif

#ifdef OPENCV_FOUND
%include "OpenCVInterop.h"
#endif
