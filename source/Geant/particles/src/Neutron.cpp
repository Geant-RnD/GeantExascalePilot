#include "Geant/particles/Neutron.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantx {
Neutron *Neutron::Definition()
{
  static Neutron instance("neutron", 2112, 3, geantx::units::kNeutronMassC2, 0);
  return &instance;
}

} // namespace geantx
