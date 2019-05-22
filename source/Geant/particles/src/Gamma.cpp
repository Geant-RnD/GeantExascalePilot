#include "Geant/particles/Gamma.hpp"

//#include "Geant/core/PhysicalConstants.hpp"
//#include "Geant/core/SystemOfUnits.hpp"

namespace geantx {
Gamma *Gamma::Definition()
{
  static Gamma instance("gamma", 22, 42, 0.0, 0.0);
  return &instance;
}

} // namespace geantx
