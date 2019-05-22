//===----------------------------------------------------------------------===//
/**
 * @file StepperFactory.h
 * @brief  Abstract field class for Geant-V prototype
 * @author John Apostolakis
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <ostream>

// Base types
#include "Geant/magneticfield/VScalarEquationOfMotion.hpp"
#include "Geant/magneticfield/VScalarIntegrationStepper.hpp"

// Concrete Types being created
#include "Geant/magneticfield/GUTCashKarpRKF45.hpp"
#include "Geant/magneticfield/TClassicalRK4.hpp"
#include "Geant/magneticfield/TSimpleRunge.hpp"

// namespace vecFieldPropagation {

/**
 * @brief Class StepperFactory
 */

class StepperFactory {
public:
  static const unsigned int Nposmom   = 6; // Position 3-vec + Momentum 3-vec
  static const int DefaultStepperCode = 5; // Cash Karp

  /**
   * @brief Track parametrized constructor
   *
   * @param EquationType - Type of Equation of Motion
   * @param equation     - Instance of Equaiton of Motion (type: EquationType)
   * @param StepperCode  - Integer Code to identify type of Stepper
   */
  template <typename EquationType>
  static VScalarIntegrationStepper *CreateStepper(EquationType *equation,
                                                  int StepperCode = DefaultStepperCode,
                                                  bool verbose    = false);

  // static StepperFactory* Instance();
};

template <typename EquationType>
VScalarIntegrationStepper *StepperFactory::CreateStepper(EquationType *equation,
                                                         int StepperCode, bool verbose)
{
  VScalarIntegrationStepper *stepper; // , *exactStepper;

  const char *stepperName        = 0;
  const char NameSimpleRunge[]   = "TSimpleRunge";
  const char NameClassicalRK4[]  = "TClassicalRK4";
  const char NameCashKarpRKF45[] = "TCashKarpRKF45";

  int MaxStepperCode = 5;

  if ((StepperCode <= 0) || (StepperCode > MaxStepperCode) ||
      (StepperCode == 2) // Missing in range  min - max
      || (StepperCode == 3))
    StepperCode = DefaultStepperCode;

  switch (StepperCode) {
  case 1:
    stepper     = new TSimpleRunge<EquationType, Nposmom>(equation);
    stepperName = NameSimpleRunge;
    break;
  // case 2: stepper = new G4SimpleHeum(equation);   break;
  // case 3: stepper = new BogackiShampine23(equation); break;
  case 4:
    stepper     = new TClassicalRK4<EquationType, Nposmom>(equation);
    stepperName = NameClassicalRK4;
    break;
  case 5:
    stepper     = new GUTCashKarpRKF45<EquationType, Nposmom>(equation);
    stepperName = NameCashKarpRKF45;
    break;
  default:
    stepper = (VScalarIntegrationStepper *)0;
    std::cerr << " ERROR> StepperFactory: No stepper selected. " << std::endl;
    break;
    // exit(1);
  }
  if (stepperName && verbose)
    std::cout << "StepperFactory: Chosen the  " << stepperName << " stepper."
              << std::endl;

  return stepper;
}

// } // end of namespace vecFieldPropagation
