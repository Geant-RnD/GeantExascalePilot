
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store pi+ static properties.
 * @class   PionPlus
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class PionPlus : public Particle {
public:
  static PionPlus *Definition();

  // copy CTR and assignment operators are deleted
  PionPlus(const PionPlus &) = delete;
  PionPlus &operator=(const PionPlus &) = delete;

private:
  PionPlus(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

} // namespace geantx
