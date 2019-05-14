#include "Geant/particles/Electron.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {
Electron *Electron::Definition()
{
  static Electron instance("e-", 11, 22, geantx::units::kElectronMassC2, -1.0 * geantx::units::eplus);
  return &instance;
}

} // namespace geantphysics
