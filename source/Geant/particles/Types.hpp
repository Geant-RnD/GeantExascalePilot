//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/particles/Types.hpp
 * @brief Declaration of particle types
 */
//===----------------------------------------------------------------------===//

#pragma once

namespace geantx {
class Particle;        // base class
class GenericParticle; // unspecified particle (for default impl)
class Geantino;        // non-interacting particle
class ChargedGeantino; // EM-field interacting particle
class Electron;
class Gamma;
class KaonLong;
class KaonMinus;
class KaonPlus;
class KaonShort;
class KaonZero;
class Neutron;
class PionMinus;
class PionPlus;
class PionZero;
class Positron;
class Proton;
} // namespace geantx
