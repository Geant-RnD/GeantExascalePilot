#include "Geant/particles/PionPlus.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantx {
PionPlus *PionPlus::Definition()
{
  static PionPlus instance("pi+", 211, 10, 0.13957 * geantx::units::GeV,
                           geantx::units::eplus); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantx
