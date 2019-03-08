#include "Geant/particles/KaonZero.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {

KaonZero *KaonZero::Definition()
{
  static KaonZero instance("K0", 311, 17, 0.497614 * geant::units::GeV, 0); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
