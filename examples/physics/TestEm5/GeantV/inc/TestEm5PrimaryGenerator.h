
#ifndef USERPRIMARYGENERATOR_H
#define USERPRIMARYGENERATOR_H

// from GeantV
#include "Geant/PrimaryGenerator.h"

// for GEANT_IMPL_NAMESPACE
#include "Geant/Typedefs.h"

namespace GEANT_IMPL_NAMESPACE {
namespace geant {
class Track;
class TaskData;
class EventInfo;
}
}

// geantphysics classdef
namespace geantphysics {
class Particle;
}

#include <string>

namespace userapplication {

/**
 * @brief User primary generator for TeatEm5.
 *
 * The primary generator is a simple particle gun with configurable primary particle type and kinetic energy.
 *
 * @class   TestEm5PrimaryGenerator
 * @author  M Novak
 * @date    July 2017
 */

class TestEm5DetectorConstruction;

class TestEm5PrimaryGenerator : public geant::PrimaryGenerator {
public:
  // CTR DTR
  TestEm5PrimaryGenerator(const TestEm5DetectorConstruction *det);
  ~TestEm5PrimaryGenerator();

  // public setters/getters
  void SetPrimaryParticleName(std::string pname) { fPrimaryParticleName = pname; }
  const std::string &GetPrimaryParticleName() const { return fPrimaryParticleName; }

  void SetPrimaryParticleEnergy(double ekin) { fPrimaryEkin = ekin; }
  double GetPrimaryParticleEnergy() const { return fPrimaryEkin; }

  void SetNumberOfPrimaryParticlePerEvent(int val) { fPrimaryPerEvent = val; }
  int GetNumberOfPrimaryParticlePerEvent() const { return fPrimaryPerEvent; }

  const geantphysics::Particle *GetPrimaryParticle() const { return fParticle; }

  // interface methods
  virtual void InitPrimaryGenerator();
  virtual geant::EventInfo NextEvent(geant::TaskData *td);
  virtual void GetTrack(int n, geant::Track &gtrack, geant::TaskData *td);

private:
  TestEm5PrimaryGenerator()                                = delete;
  TestEm5PrimaryGenerator(const TestEm5PrimaryGenerator &) = delete;
  TestEm5PrimaryGenerator &operator=(const TestEm5PrimaryGenerator &) = delete;

private:
  std::string fPrimaryParticleName; // name of the primary particle
  int fPrimaryPerEvent;             // number of primary particle to be generated per event
  int fPDG;                         // PDG code of parimary particles
  int fGVPartIndex;                 // internal GV particle index of the primary
  double fPrimaryEkin;              // kinetic energy of the primary in internal [energy] unit
  double fXPos;                     // (x,y,z) position of the primary particles in internal [length] unit
  double fYPos;
  double fZPos;
  double fXDir; // direction vector of the primary particles
  double fYDir;
  double fZDir;
  //
  double fMass;   // rest mass of the primary in internal [energy] unit
  double fCharge; // charge of the primary in internal [charge] unit
  double fETotal; // total energy of the primary in internal [energy] unit
  double fPTotal; // total momentum of the primary in internal [energy] unit
  //
  const geantphysics::Particle *fParticle;      // the primary particle
  const TestEm5DetectorConstruction *fDetector; // the detector
};

} // namespace userapplication

#endif // USERPRIMARYGENERATOR_H
