#include "Geant/particles/Positron.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
Positron *Positron::Definition()
{
  static Positron instance("e+", -11, 23, geantx::units::kElectronMassC2, 1.0 * geantx::units::eplus);
  return &instance;
}

} // namespace geantphysics
