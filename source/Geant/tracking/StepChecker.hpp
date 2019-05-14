//
//
//   Author: J. Apostolakis, 28 Jan 2016

#pragma once

#include "base/Vector3D.h"
#include "Geant/core/Config.hpp"
#include "Geant/core/Typedefs.hpp"

namespace geantx {

class StepChecker {
public:
  StepChecker(double eps, double maxLengthDiff = 0.0, bool verbose = false);
  ~StepChecker() {}

  /** @brief Compare two different 'solutions' - each provides a endpoint
   * position/direction */
  bool CompareStep(ThreeVector const &position, ThreeVector const &direction,
                   double charge, double momentum, double step,
                   ThreeVector const &endPosition1, ThreeVector const &endDirection1,
                   ThreeVector const &endPositionRef, ThreeVector const &endDirectionRef) const;

  /** @brief Check against helical solution */
  bool CheckStep(ThreeVector const &position, ThreeVector const &direction,
                 double charge, double momentum, double step,
                 ThreeVector const &endPosition,
                 ThreeVector const &endDirection, ThreeVector const &BfieldVec) const;

  // Potential to generalise methods above using template:
  // template<typename VectorType, typename BaseFltType, typename BaseIntType>
  //  in place of i) ThreeVector , ii) double,  iii) int
private:
  double fMaxRelDiff;  /** max relative difference accepted - displacement or direction
                        */
  double fMaxLengthDf; /** max displacement accepted without warning / error */
  bool fVerbose;       /** Print comparison or not */
};

inline StepChecker::StepChecker(double eps, double maxLengthDiff, bool verbose)
{
  fMaxRelDiff  = eps;
  fMaxLengthDf = maxLengthDiff;
  fVerbose     = verbose;
}

} // namespace geantx
