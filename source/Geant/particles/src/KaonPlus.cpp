#include "Geant/particles/KaonPlus.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
KaonPlus *KaonPlus::Definition()
{
  static KaonPlus instance("K+", 321, 13, 0.493677 * geantx::units::GeV,
                           geantx::units::eplus); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
