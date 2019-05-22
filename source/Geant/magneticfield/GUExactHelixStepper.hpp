//
//  GUExactHelixStepper
//  -------------------
//
//  Based on G4ExactHelixStepper
//
// Adapted from G4ExactHelixStepper
// - 16.Oct.15  J.Apostolakis   Adapted
// --------------------------------------------------------------------

#pragma once

#include "Geant/core/Typedefs.hpp"
#include "Geant/magneticfield/ScalarMagFieldEquation.hpp"
#include "Geant/magneticfield/VScalarHelicalStepper.hpp"
#include <base/Vector3D.h>

class GUExactHelixStepper : public VScalarHelicalStepper {
public:                                                // with description
  GUExactHelixStepper(VScalarEquationOfMotion *EqRhs); // ScalarMagFieldEquation *EqRhs);
  ~GUExactHelixStepper();

  void StepWithErrorEstimate(const double y[], const double dydx[], double charge,
                             double h, double yout[], double yerr[]) override final;
  // Step 'integration' for step size 'h'
  // Provides helix starting at y[0 to 6]
  // Outputs yout[] and ZERO estimated error yerr[]=0.

  void StepWithoutErrorEstimate(const double y[], ThreeVector Bfld, double charge,
                                double h, double yout[]) override final;
  // Performs a 'dump' Step without error calculation.

  double DistChord(double charge) const override final;
  // Estimate maximum distance of curved solution and chord ...

private:
  GUExactHelixStepper(const GUExactHelixStepper &);
  GUExactHelixStepper &operator=(const GUExactHelixStepper &);
  // Private copy constructor and assignment operator.

private:
  // ThreeVector    fBfieldValue;
  vecgeom::Vector3D<double> fBfieldValue;
  //  Initial value of field at last step

  // ScalarMagFieldEquation*
  VScalarEquationOfMotion *fPtrMagEqOfMot;
};
