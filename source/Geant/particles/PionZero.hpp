
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store pi0 static properties.
 * @class   PionZero
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class PionZero : public Particle {
public:
  static PionZero *Definition();

  // copy CTR and assignment operators are deleted
  PionZero(const PionZero &) = delete;
  PionZero &operator=(const PionZero &) = delete;

private:
  PionZero(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {
  }
};

} // namespace geantx
