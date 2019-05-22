//
//
// Derived from G4MagIntegrationStepper class
//
// --------------------------------------------------------------------

#include "Geant/magneticfield/VScalarIntegrationStepper.hpp"

// Constructor for stepper abstract base class.
//

VScalarIntegrationStepper::VScalarIntegrationStepper(VScalarEquationOfMotion *equation,
                                                     unsigned int integrationOrder,
                                                     unsigned int num_integration_vars,
                                                     int num_state_vars)
    : fAbstrEquation(equation), fIntegrationOrder(integrationOrder),
      fNoIntegrationVariables(num_integration_vars),
      fNoStateVariables(num_state_vars > 0 ? num_state_vars : num_integration_vars)
{}

VScalarIntegrationStepper::~VScalarIntegrationStepper() {}

// This allows the method to cache the value etc - Not needed for now
// void VScalarIntegrationStepper::ComputeRightHandSide( const double y[], double charge,
// double dydx[] )
// {
//    this->RightHandSide( y, charge, dydx );
// }

void VScalarIntegrationStepper::SetEquationOfMotion(VScalarEquationOfMotion *newEquation)
{
  if (newEquation != nullptr) {
    fAbstrEquation = newEquation;
  }
}
