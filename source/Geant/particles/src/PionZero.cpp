#include "Geant/particles/PionZero.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
PionZero *PionZero::Definition()
{
  static PionZero instance("pi0", 111, 12, 0.134977 * geantx::units::GeV,
                           0); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
