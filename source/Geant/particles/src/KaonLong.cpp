#include "Geant/particles/KaonLong.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
KaonLong *KaonLong::Definition()
{
  static KaonLong instance("K_L0", 130, 15, 0.497614 * geantx::units::GeV, 0); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
