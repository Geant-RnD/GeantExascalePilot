#include "Geant/particles/PionMinus.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
PionMinus *PionMinus::Definition()
{
  static PionMinus instance(
      "pi-", -211, 11, 0.13957 * geantx::units::GeV,
      -1.0 * geantx::units::eplus); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
