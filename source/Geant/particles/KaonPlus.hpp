
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store K+ static properties.
 * @class   KaonPlus
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class KaonPlus : public Particle {
public:
  static KaonPlus *Definition();

  // copy CTR and assignment operators are deleted
  KaonPlus(const KaonPlus &) = delete;
  KaonPlus &operator=(const KaonPlus &) = delete;

private:
  KaonPlus(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {
  }
};

} // namespace geantx
