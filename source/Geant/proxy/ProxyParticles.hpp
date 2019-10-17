//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
//
/**
 * @file
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/particles/Particle.hpp"
#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

namespace geantx {

//======================================================================================//

class CpuGamma : public Particle {
public:
  static CpuGamma *Definition()
  {
    static CpuGamma instance("CpuGamma", 0, 0, 0.0, 0.0);
    return &instance;
  }

  // copy CTR and assignment operators are deleted
  CpuGamma(const CpuGamma &) = delete;
  CpuGamma &operator=(const CpuGamma &) = delete;

private:
  CpuGamma(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

//======================================================================================//

class CpuElectron : public Particle {
public:
  static CpuElectron *Definition()
  {
    static CpuElectron instance("CpuElectron", 1, 1, geantx::units::kElectronMassC2,
                                -1.0 * geantx::units::eplus);
    return &instance;
  }

  // copy CTR and assignment operators are deleted
  CpuElectron(const CpuElectron &) = delete;
  CpuElectron &operator=(const CpuElectron &) = delete;

private:
  CpuElectron(const std::string &name, int pdgcode, int intcode, double mass,
              double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

//======================================================================================//

class GpuGamma : public Particle {
public:
  static GpuGamma *Definition()
  {
    static GpuGamma instance("GpuGamma", 0, 0, 0.0, 0.0);
    return &instance;
  }

  // copy CTR and assignment operators are deleted
  GpuGamma(const GpuGamma &) = delete;
  GpuGamma &operator=(const GpuGamma &) = delete;

private:
  GpuGamma(const std::string &name, int pdgcode, int intcode, double mass, double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

//======================================================================================//

class GpuElectron : public Particle {
public:
  static GpuElectron *Definition()
  {
    static GpuElectron instance("GpuElectron", 1, 1, geantx::units::kElectronMassC2,
                                -1.0 * geantx::units::eplus);
    return &instance;
  }

  // copy CTR and assignment operators are deleted
  GpuElectron(const GpuElectron &) = delete;
  GpuElectron &operator=(const GpuElectron &) = delete;

private:
  GpuElectron(const std::string &name, int pdgcode, int intcode, double mass,
              double charge)
      : Particle(name, pdgcode, intcode, mass, charge)
  {}
};

//======================================================================================//

} // namespace geantx
