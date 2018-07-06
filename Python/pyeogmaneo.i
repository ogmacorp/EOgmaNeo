// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
#ifdef BUILD_PREENCODERS
#include "KMeansEncoder.h"
#include "ImageEncoder.h"
#include "GaborEncoder.h"
#endif
%}

// Handle STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%template(StdVeci) std::vector<int>;
%template(Std2DVeci) std::vector<std::vector<int> >;
%template(StdPairi) std::pair<int, int>;
%template(StdPairf) std::pair<float, float>;
%template(StdVecPairi) std::vector<std::pair<int, int> >;
%template(StdVecLayerDesc) std::vector<eogmaneo::LayerDesc>;
%template(StdVecf) std::vector<float>;
%template(Std2DVecf) std::vector<std::vector<float> >;
%template(StdVecb) std::vector<bool>;

%ignore eogmaneo::LayerForwardWorkItem;
%ignore eogmaneo::LayerBackwardWorkItem;

%include "ComputeSystem.h"
%include "Layer.h"
%include "Hierarchy.h"
#ifdef BUILD_PREENCODERS
%include "KMeansEncoder.h"
%include "ImageEncoder.h"
%include "GaborEncoder.h"
#endif
