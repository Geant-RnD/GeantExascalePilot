/// \file vc/backend.h

#ifndef VECGEOM_BACKEND_VCFLOATBACKEND_H_
#define VECGEOM_BACKEND_VCFLOATBACKEND_H_

// #include "base/Global.h"
#include "base/Global.h"

#include "backend/scalar/Backend.h"

#include <Vc/Vc>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

struct kVcFloat {
  typedef Vc::int_v int_v;
  typedef Vc::Vector<float> precision_v;
  typedef Vc::Vector<float>::Mask bool_v;
  typedef Vc::Vector<int> inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  // alternative typedefs ( might supercede above typedefs )
  typedef Vc::int_v Int_t;
  typedef Vc::Vector<Precision> Double_t;
  typedef Vc::Vector<Precision>::Mask Bool_t;
  typedef Vc::Vector<Precision> Index_t;
};

// constexpr int kVectorSize = kVcFloat::precision_v::Size;
typedef kVcFloat::int_v VcIntFloat;
typedef kVcFloat::precision_v VcPrecisionFloat;
typedef kVcFloat::bool_v VcBoolFloat;
typedef kVcFloat::inside_v VcInsideFloat;

#if 0
// The VcFloat backend is included at the same type as Vc backend, consequently
// this can not be used ...
constexpr int kVectorSize = kVcFloat::precision_v::Size;
#define VECGEOM_BACKEND_TYPE vecgeom::kVcFloat
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) vecgeom::kVcFloat::VcPrecision(P)
#define VECGEOM_BACKEND_PRECISION_TYPE vecgeom::kVcFloat::VcPrecision
#define VECGEOM_BACKEND_PRECISION_NOT_SCALAR
#define VECGEOM_BACKEND_BOOL vecgeom::kVcFloat::VcBoolFloat
#define VECGEOM_BACKEND_INSIDE vecgeom::kVcFloat::inside_v
#endif

// template <typename Type>
// VECGEOM_FORCE_INLINE
// void CondAssign(typename Vc::Vector<Type>::Mask const &cond,
//                Type const &thenval,
//                Type const &elseval,
//                Vc::Vector<Type> *const output) {
//  (*output)(cond) = thenval;
//  (*output)(!cond) = elseval;
//}
//
// template <typename Type>
// VECGEOM_FORCE_INLINE
// void MaskedAssign(typename Vc::Vector<Type>::Mask const &cond,
//                  Vc::Vector<Type> const &thenval,
//                  Vc::Vector<Type> *const output) {
//  (*output)(cond) = thenval;
//}
//
// template <typename Type>
// VECGEOM_FORCE_INLINE
// void MaskedAssign(typename Vc::Vector<Type>::Mask const &cond,
//                  Type const &thenval,
//                  Vc::Vector<Type> *const output) {
//  (*output)(cond) = thenval;
//}

// in case of scalar VcBoolFloat == VcBoolDouble == bool
#ifndef VC_IMPL_Scalar
VECGEOM_FORCE_INLINE
void MaskedAssign(VcBoolFloat const &cond, const Inside_t thenval, VcInside *const output)
{
  (*output)(Vc::simd_cast<VcInside::Mask>(cond)) = thenval;
}

VECGEOM_FORCE_INLINE
bool IsFull(VcBoolFloat const &cond)
{
  return cond.isFull();
}

VECGEOM_FORCE_INLINE
bool Any(VcBoolFloat const &cond)
{
  return !cond.isEmpty();
}

VECGEOM_FORCE_INLINE
bool IsEmpty(VcBoolFloat const &cond)
{
  return cond.isEmpty();
}
#endif

VECGEOM_FORCE_INLINE
VcPrecisionFloat Abs(VcPrecisionFloat const &val)
{
  return Vc::abs(val);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat Sqrt(VcPrecisionFloat const &val)
{
  return Vc::sqrt(val);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat ATan2(VcPrecisionFloat const &y, VcPrecisionFloat const &x)
{
  return Vc::atan2(y, x);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat sin(VcPrecisionFloat const &x)
{
  return Vc::sin(x);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat cos(VcPrecisionFloat const &x)
{
  return Vc::cos(x);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat tan(VcPrecisionFloat const &radians)
{
  // apparently Vc does not have a tan function
  //  return Vc::tan(radians);
  // emulating it for the moment
  VcPrecisionFloat s, c;
  Vc::sincos(radians, &s, &c);
  return s / c;
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat Min(VcPrecisionFloat const &val1, VcPrecisionFloat const &val2)
{
  return Vc::min(val1, val2);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat Max(VcPrecisionFloat const &val1, VcPrecisionFloat const &val2)
{
  return Vc::max(val1, val2);
}

VECGEOM_FORCE_INLINE
VcPrecisionFloat Floor(VcPrecisionFloat const &val)
{
  return Vc::floor(val);
}

} // End inline namespace

} // End global namespace

#endif // VECGEOM_BACKEND_VCBACKEND_H_
