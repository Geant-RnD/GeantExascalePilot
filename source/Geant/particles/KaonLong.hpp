
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store K_L0 static properties.
 * @class   KaonLong
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class KaonLong : public Particle {
public:
  static KaonLong *Definition();

  // copy CTR and assignment operators are deleted
  KaonLong(const KaonLong &) = delete;
  KaonLong &operator=(const KaonLong &) = delete;

private:
  KaonLong(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {
  }
};

} // namespace geantx
