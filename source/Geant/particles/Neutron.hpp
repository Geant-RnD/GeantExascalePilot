
#pragma once

#include "Geant/particles/Particle.hpp"

namespace geantx {
/**
 * @brief   Class(singletone) to store neutron static properties.
 * @class   Neutron
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class Neutron : public Particle {
public:
  static Neutron *Definition();

  // copy CTR and assignment operators are deleted
  Neutron(const Neutron &) = delete;
  Neutron &operator=(const Neutron &) = delete;

private:
  Neutron(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

} // namespace geantx
