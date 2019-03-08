
#include "TestEm3PrimaryGenerator.h"
#include "TestEm3DetectorConstruction.h"
#include "Geant/Particle.h"

// from geantV
#include "Geant/Track.h"

#include <cmath>

namespace userapplication {

TestEm3PrimaryGenerator::TestEm3PrimaryGenerator(TestEm3DetectorConstruction *det) : fDetector(det)
{
  fPrimaryParticleName = "e-";
  fPrimaryPerEvent     = 1;
  fParticle            = nullptr;
  //
  fGVPartIndex = 0;
  //
  fPrimaryEkin = 10. * geant::units::GeV;
  //
  fXPos = 0.;
  fYPos = 0.;
  fZPos = 0.;
  //
  fXDir = 1.;
  fYDir = 0.;
  fZDir = 0.;
  //
  fMass   = 0.;
  fCharge = 0.;
  fETotal = 0.;
  fPTotal = 0.;
}

TestEm3PrimaryGenerator::~TestEm3PrimaryGenerator() {}

void TestEm3PrimaryGenerator::InitPrimaryGenerator()
{
  fParticle = geantphysics::Particle::GetParticleByName(fPrimaryParticleName);
  if (!fParticle) {
    std::cerr << "   \n *** ERROR::UserPrimaryGenerator::InitPrimaryGenerator()    \n"
              << "          unknown particle name = " << fPrimaryParticleName << " \n"
              << std::endl;
    exit(-1);
  }
  fGVPartIndex = fParticle->GetInternalCode();
  fMass        = fParticle->GetPDGMass();
  fCharge      = fParticle->GetPDGCharge();
  fETotal      = fPrimaryEkin + fMass;
  fPTotal      = std::sqrt((fETotal - fMass) * (fETotal + fMass));
  //
  fXPos = -0.25 * (fDetector->GetWorldSizeX() + fDetector->GetCaloSizeX());
  fYPos = 0.;
  fZPos = 0.;
  //
  fXDir = 1.;
  fYDir = 0.;
  fZDir = 0.;
}

geant::EventInfo TestEm3PrimaryGenerator::NextEvent(geant::TaskData * /*td*/)
{
  geant::EventInfo current;
  current.ntracks = fPrimaryPerEvent;
  current.xvert   = fXPos;
  current.yvert   = fYPos;
  current.zvert   = fZPos;
  return current;
}

void TestEm3PrimaryGenerator::GetTrack(int /*n*/, geant::Track &gtrack, geant::TaskData * /*td*/)
{
  gtrack.SetGVcode(fGVPartIndex);
  gtrack.SetPosition(fXPos, fYPos, fZPos);
  gtrack.SetDirection(fXDir, fYDir, fZDir);
  //
  gtrack.SetCharge(fCharge);
  gtrack.SetMass(fMass);
  gtrack.SetE(fETotal);
  gtrack.SetP(fPTotal);
}

} // namespace userapplication
