//===----------------------------------------------------------------------===//
/**
 * @file VVectorField.h
 * @brief  Abstract field class for Geant-V prototype
 * @author John Apostolakis
 */
//===----------------------------------------------------------------------===//

//
//
// class VVectorField
//
// Class description:
//
// Abstract class for any kind of Field.
// It allows any kind of field (vector, scalar, tensor and any set of them)
// to be defined by implementing the inquiry function interface.
//
// The key method is  GetFieldValue( const  double Point[4],
//                    *************         double *fieldArr )
// Given an input position/time vector 'Point',
// this method must return the value of the field in "fieldArr".
//
// A field must also specify whether it changes a track's energy:
//                    DoesFieldChangeEnergy()
//                    *********************
// A field must co-work with a corresponding Equation of Motion, to
// enable the integration of a particle's position, momentum and, optionally,
// spin.  For this a field and its equation of motion must follow the
// same convention for the order of field components in the array "fieldArr"
// -------------------------------------------------------------------

#pragma once

#include "base/Global.h"
#include "base/Vector3D.h"
#include <Geant/core/VectorTypes.hpp>
#include <vector>

// #include "GUVTypes.hh"
// #include "globals.hh"

/**
 * @brief Base class describing the scalar and vector interfaces for Field classes
 */

class VVectorField {
public: // with description
  using Double_v = geantx::Double_v;
  using Float_v  = geantx::Float_v;

  template <typename T>
  using Vector3D = vecgeom::Vector3D<T>;

  /**
   * @brief Scalar interface for field retrieval
   *
   * @param Position - position (0,1,2=x,y,z)   [Input]   - Note: time is suppressed => B(t)=B(0)
   * @param fieldArr - output values of field. Usual convention:
   *                   0,1,2 = B_x, B_y, B_z
   *                   3,4,5 = E_x, E_y, E_z  (foreseen extension)
   *        Units are expected to be native GeantV units.
   */
  virtual void ObtainFieldValue(const Vector3D<double> &position, Vector3D<double> &fieldValue) = 0;

  /** @brief Vector interface for field retrieval */
  virtual void ObtainFieldValueSIMD(const Vector3D<Double_v> &position, Vector3D<Double_v> &fieldValue) = 0;

  inline VVectorField(int numberOfComponents, bool changesEnergy)
      : fNumberOfComponents(numberOfComponents), fChangesEnergy(changesEnergy)
  {
  }

  inline VVectorField(const VVectorField &field)
      : fNumberOfComponents(field.fNumberOfComponents), fChangesEnergy(field.fChangesEnergy)
  {
  }
  virtual ~VVectorField() {}

  // A field signature function that can be used to insure
  // that the Equation of motion object and the VVectorField object
  // have the same "field signature"?

  bool DoesFieldChangeEnergy() const { return fChangesEnergy; }
  int GetNumberOfComponents() const { return fNumberOfComponents; }

  VVectorField &operator=(const VVectorField &field)
  {
    if (&field != this) {
      fNumberOfComponents = field.fNumberOfComponents;
      fChangesEnergy      = field.fChangesEnergy;
    }
    return *this;
  }

  virtual VVectorField *Clone() const
  {
    std::runtime_error("Clone must be implemented by the derived field class");
    return nullptr;
    // Implements cloning, likely needed for MT
  }

  // Expect a method of the following signature
  //  [Derived-Field-type] * CloneOrSafeSelf( bool* pSafe ) const
  // to be implemented for each derived class.
  // If the class is thread-safet, it can be implemented as:
  //  { if( pSafe ) { *pSafe= false; } ; return Clone(); }

private:
  int fNumberOfComponents; // E.g.  B -> N=3 , ie x,y,z
                           //       E+B -> N=6
  bool fChangesEnergy;     // Electric: true, Magnetic: false
};
