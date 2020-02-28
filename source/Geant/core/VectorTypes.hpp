//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Declaration of the vector (instructions) types.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <Geant/core/Config.hpp>
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
using VectorBackend = vecgeom::VectorBackend;
using Float_v       = VectorBackend::Float_v;
using Double_v      = VectorBackend::Double_v;
using Int_v         = VectorBackend::Int_v;

using MaskF_v = vecCore::Mask_v<Float_v>;
using MaskD_v = vecCore::Mask_v<Double_v>;
using MaskI_v = vecCore::Mask_v<Int_v>;

using IndexD_v = vecCore::Index<Double_v>;
using IndexF_v = vecCore::Index<Float_v>;

const int kVecLenF = vecCore::VectorSize<Float_v>();
const int kVecLenD = vecCore::VectorSize<Double_v>();

const auto kVecAlignD = sizeof(Double_v);
const auto kVecAlignF = sizeof(Float_v);

GEANT_FORCE_INLINE
void CopyFltToDbl(Float_v const &flt_v, Double_v &dbl1_v, Double_v &dbl2_v)
{
  // Copy the float SIMD lanes into 2 Double_v variables
  for (size_t lane = 0; lane < kVecLenD; ++lane) {
    vecCore::Set(dbl1_v, lane, (double)vecCore::Get(flt_v, lane));
    vecCore::Set(dbl2_v, lane, (double)vecCore::Get(flt_v, lane + kVecLenD));
  }
}

GEANT_FORCE_INLINE
void CopyFltToDbl(vecgeom::Vector3D<Float_v> const &flt_v,
                  vecgeom::Vector3D<Double_v> &dbl1_v,
                  vecgeom::Vector3D<Double_v> &dbl2_v)
{
  // Copy the float SIMD lanes into 2 Double_v variables
  for (size_t lane = 0; lane < kVecLenD; ++lane) {
    for (size_t i = 0; i < 3; ++i) {
      vecCore::Set(dbl1_v[i], lane, (double)vecCore::Get(flt_v[i], lane));
      vecCore::Set(dbl2_v[i], lane, (double)vecCore::Get(flt_v[i], lane + kVecLenD));
    }
  }
}

GEANT_FORCE_INLINE
void CopyDblToFlt(Double_v const &dbl1_v, Double_v const &dbl2_v, Float_v &flt_v)
{
  // Copy the 2 Double_v SIMD lanes into one Float_v variable
  for (size_t lane = 0; lane < kVecLenD; ++lane) {
    vecCore::Set(flt_v, lane, (float)vecCore::Get(dbl1_v, lane));
    vecCore::Set(flt_v, lane + kVecLenD, (double)vecCore::Get(dbl2_v, lane));
  }
}

GEANT_FORCE_INLINE
void CopyDblToFlt(vecgeom::Vector3D<Double_v> const &dbl1_v,
                  vecgeom::Vector3D<Double_v> const &dbl2_v,
                  vecgeom::Vector3D<Float_v> &flt_v)
{
  // Copy the 2 Double_v SIMD lanes into one Float_v variable
  for (size_t lane = 0; lane < kVecLenD; ++lane) {
    for (size_t i = 0; i < 3; ++i) {
      vecCore::Set(flt_v[i], lane, (float)vecCore::Get(dbl1_v[i], lane));
      vecCore::Set(flt_v[i], lane + kVecLenD, (double)vecCore::Get(dbl2_v[i], lane));
    }
  }
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
