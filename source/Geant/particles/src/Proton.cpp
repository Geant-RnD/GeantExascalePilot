#include "Geant/particles/Proton.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
Proton *Proton::Definition()
{
  static Proton instance("proton", 2212, 1, geant::units::kProtonMassC2, 1.0 * geant::units::eplus);
  return &instance;
}

} // namespace geantphysics
