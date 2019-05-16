#include "Geant/particles/KaonShort.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantx {
KaonShort *KaonShort::Definition()
{
  static KaonShort instance("K_S0", 310, 16, 0.497614 * geantx::units::GeV,
                            0); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantx
