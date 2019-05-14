#pragma once

#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackState.hpp"
#include "Geant/track/ParticleDefinitions.hpp"

namespace geantx {

//---------------------------------------------------------------------------//
/*!
 * \class TrackPhysicsAccessor
 * \brief Abstract physics state information access for a track.
 *
 * This currently takes just a track state, but if the data layout changes, it
 * could take e.g. some abstract multi-particle state *plus* a particle index.
 */
template <class PD>
class TrackPhysicsAccessor : public TrackAccessor {
  using Base = TrackAccessor;

public:
  using ParticleDef_t = PD;

private:
  const ParticleDef_t &fParDef;

public:
  TrackPhysicsAccessor(const TrackCollection &tracks, TrackId_t track_id, const ParticleDefinitions &pdefs)
      : Base(tracks, track_id), fParDef(GetParticleDef(pdefs, Base::State()))
  {
  }

  // >>> ACCESSORS

  double Charge() const { return fParDef.Charge(); }
  Species_t Species() const { return fParDef.Species(); }
  //! Rest mass (* c^2: same units as energy, since c=1)
  double Mass() const { return fParDef.fMass(); }

  //! Momentum
  double P() const { return this->Pstate().fMomentum; }

  //! Kinetic energy
  double Ekin() const { return this->Pstate().fEkin; }

  //! Total (rest mass + kinetic) energy
  double E() const { return this->Mass() + this->Ekin(); }

  //! Natural logarithm of kinetic energy
  double LogEkin() const {
    return std::log(this->Ekin());
  }

  MaterialId_t Material() const { return this->State().fMaterialState.fMaterial; }

protected:

  // >>> IMPLEMENTATION DETAILS

  static const ParticleDef_t &GetParticleDef(const ParticleDefinitions &pdefs, const TrackState &state)
  {
      return *static_cast<const ParticleDef_t *>(pdefs.Get(state.fPhysicsState.fParticleDefId));
  }


  TrackPhysicsAccessor(const TrackState &track, const ParticleDefinitions &pdefs)
      : Base(track), fParDef(GetParticleDef(pdefs, Base::State()))
  {
  }

  TrackPhysicsAccessor(const TrackState &track, const ParticleDef_t &type)
      : Base(track), fParDef(type)
  {
  }

  const TrackPhysicsState &Pstate() const { return this->State().fPhysicsState; }

  template<class PT> friend class TrackModifier;
};

} // namespace geantx
