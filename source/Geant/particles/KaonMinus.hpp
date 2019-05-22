
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store K- static properties.
 * @class   KaonMinus
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class KaonMinus : public Particle {
public:
  static KaonMinus *Definition();

  // copy CTR and assignment operators are deleted
  KaonMinus(const KaonMinus &) = delete;
  KaonMinus &operator=(const KaonMinus &) = delete;

private:
  KaonMinus(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

} // namespace geantx
