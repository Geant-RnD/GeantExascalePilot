
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store K_S0 static properties.
 * @class   KaonShort
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class KaonShort : public Particle {
public:
  static KaonShort *Definition();

  // copy CTR and assignment operators are deleted
  KaonShort(const KaonShort &) = delete;
  KaonShort &operator=(const KaonShort &) = delete;

private:
  KaonShort(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {
  }
};

} // namespace geantx
