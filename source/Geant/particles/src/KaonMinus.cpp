#include "Geant/particles/KaonMinus.hpp"

#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantphysics {

KaonMinus *KaonMinus::Definition()
{
  static KaonMinus instance("K-", -321, 14, 0.493677 * geant::units::GeV,
                            -1.0 * geant::units::eplus); // mass value taken from Geant4 10.3
  return &instance;
}

} // namespace geantphysics
