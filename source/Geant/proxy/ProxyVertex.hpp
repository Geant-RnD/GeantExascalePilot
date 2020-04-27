//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyVertex.hpp
 * @brief vertex of primary particles
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/Types.hpp"
#include "Geant/proxy/ProxyVector.cuh"

namespace geantx {

class ProxyVertex {

  using Particle_t = TrackState;
  using ParticleVector = ProxyVector<Particle_t*>;

public:

  ProxyVertex() : fPDG(0), fTime(0), fPosition(0,0,0)
  { fParticleVector = new ParticleVector(); }

  ProxyVertex(double t0, ThreeVector x0) : fPDG(0), fTime(t0), fPosition(x0)
  { fParticleVector = new ParticleVector(); }

  ~ProxyVertex() { fParticleVector->clear(); }

  int GetPDG() const { return fPDG; }

  int GetNumberOfParticles() const { return fParticleVector->size(); }

  double GetTime() const { return fTime; }

  ThreeVector GetPosition() const { return fPosition; }

  ParticleVector* GetParticleVector() const { return fParticleVector; }

  Particle_t* GetParticle(int index) const 
  { 
    REQUIRE(index < GetNumberOfParticles()); 
    return fParticleVector->Get(index); 
  }

  void SetPDG(int pdf) { fPDG = pdf; }

  void SetTime(double time) { fTime = time; }

  void SetPosition(ThreeVector position) { fPosition = position; }

  void AddParticle(Particle_t* particle) { fParticleVector->push_back(particle); }

private:

  int fPDG;                         // 0 = the primary vertex (event origin);
  double fTime;
  ThreeVector fPosition;
  ParticleVector* fParticleVector;
};

} // namespace geantx
