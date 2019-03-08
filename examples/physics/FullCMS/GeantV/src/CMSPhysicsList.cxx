
#include <Geant/GSMSCModelSimplified.h>
#include "CMSPhysicsList.h"

#include "Geant/PhysicalConstants.h"
#include "Geant/SystemOfUnits.h"

#include "Geant/PhysicsProcess.h"

#include "Geant/Particle.h"
#include "Geant/Electron.h"
#include "Geant/Positron.h"
#include "Geant/Gamma.h"

#include "Geant/ElectronIonizationProcess.h"
#include "Geant/MollerBhabhaIonizationModel.h"

#include "Geant/ElectronBremsstrahlungProcess.h"
#include "Geant/SeltzerBergerBremsModel.h"
#include "Geant/RelativisticBremsModel.h"

#include "Geant/PositronAnnihilationProcess.h"

#include "Geant/ComptonScatteringProcess.h"
#include "Geant/KleinNishinaComptonModel.h"

#include "Geant/GammaConversionProcess.h"
#include "Geant/BetheHeitlerPairModel.h"
#include "Geant/RelativisticPairModel.h"

#include "Geant/GammaPhotoElectricProcess.h"
#include "Geant/SauterGavrilaPhotoElectricModel.h"

#include "Geant/MSCProcess.h"
#include "Geant/MSCModel.h"
#include "Geant/GSMSCModel.h"

#include <numeric>

namespace cmsapp {

CMSPhysicsList::CMSPhysicsList(const geant::GeantConfig &config, const std::string &name, bool useSamplingTables)
    : geantphysics::PhysicsList(name), fUseSamplingTables(useSamplingTables), fVectorized(config.fUseVectorizedPhysics),
      fVectorizedMSC(config.fUseVectorizedMSC)
{
}

CMSPhysicsList::~CMSPhysicsList() {}

void CMSPhysicsList::Initialize()
{
  // get the partcile table and loop over it
  std::vector<geantphysics::Particle *> pTable = geantphysics::Particle::GetTheParticleTable();
  for (unsigned int i = 0; i < pTable.size(); ++i) {
    geantphysics::Particle *particle = pTable[i];
    if (particle == geantphysics::Electron::Definition()) {
      //
      // create ionization process for e- with 1 model:
      //
      geantphysics::EMPhysicsProcess *eIoniProc = new geantphysics::ElectronIonizationProcess("e-Ioni");
      // create the Moller-Bhabha model for ionization i.e. for e- + e- -> e- + e- intercation
      geantphysics::EMModel *eMBModel = new geantphysics::MollerBhabhaIonizationModel(true);
      eMBModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      eMBModel->SetLowEnergyUsageLimit(1.0 * geant::units::keV);
      eMBModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      eMBModel->SetUseSamplingTables(fUseSamplingTables);
      // add the model to the process
      eIoniProc->AddModel(eMBModel);
      //
      // add the process to the e- particle
      AddProcessToParticle(particle, eIoniProc);
      //
      // create bremsstrahlung process for e- with 2 models:
      //
      geantphysics::EMPhysicsProcess *eBremProc = new geantphysics::ElectronBremsstrahlungProcess("e-Brem");
      // create a SeltzerBergerBremsModel for e-
      geantphysics::EMModel *eSBModel = new geantphysics::SeltzerBergerBremsModel(true);
      eSBModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      eSBModel->SetLowEnergyUsageLimit(1.0 * geant::units::keV);
      eSBModel->SetHighEnergyUsageLimit(1.0 * geant::units::GeV);
      eSBModel->SetUseSamplingTables(fUseSamplingTables);
      // how to inactivate this model in a given region i.e. region with index 1
      // active regions for a model are set based on their process active regions + user requested inactive regions
      // eSBModel->AddToUserRequestedInActiveRegions(1);
      //
      // add this model to the process
      eBremProc->AddModel(eSBModel);
      //
      // create a RelativisticBremsModel for e-
      geantphysics::EMModel *eRelBModel = new geantphysics::RelativisticBremsModel();
      eRelBModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      eRelBModel->SetLowEnergyUsageLimit(1.0 * geant::units::GeV);
      eRelBModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      eRelBModel->SetUseSamplingTables(fUseSamplingTables);
      // add this model to the process
      eBremProc->AddModel(eRelBModel);
      //
      // add the process to the e- particle
      AddProcessToParticle(particle, eBremProc);
      //
      // create MSC process with 2 models:
      // = 1. with range-factor = 0.2  (EM-opt1) settings and active everywhere but in HcalRegion (#11)
      // = 2. with range-factor = 0.06 (EM-opt0) settings and active only in HcalRegion (#11)
      geantphysics::EMPhysicsProcess *eMSCProc = new geantphysics::MSCProcess("e-msc");
      // create GS-msc model, set min/max usage limits
      geantphysics::MSCModel *gsMSCModelOpt1 = nullptr;
      geantphysics::MSCModel *gsMSCModelOpt0 = nullptr;
      if (fVectorizedMSC) {
        gsMSCModelOpt1 = new geantphysics::GSMSCModelSimplified();
        gsMSCModelOpt0 = new geantphysics::GSMSCModelSimplified();
      } else {
        gsMSCModelOpt1 = new geantphysics::GSMSCModel();
        gsMSCModelOpt0 = new geantphysics::GSMSCModel();
      }
      gsMSCModelOpt1->SetBasketizable(fVectorizedMSC);
      gsMSCModelOpt0->SetBasketizable(fVectorizedMSC);
      // one with opt1 and one with opt0 stepping i.e. RangeFactor = 0.2 and 0.06
      gsMSCModelOpt1->SetRangeFactor(0.2);
      gsMSCModelOpt0->SetRangeFactor(0.06);
      //
      gsMSCModelOpt1->SetMSCSteppingAlgorithm(geantphysics::MSCSteppingAlgorithm::kUseSaftey);
      gsMSCModelOpt0->SetMSCSteppingAlgorithm(geantphysics::MSCSteppingAlgorithm::kUseSaftey);
      //
      gsMSCModelOpt1->SetLowEnergyUsageLimit(100. * geant::units::eV);
      gsMSCModelOpt1->SetHighEnergyUsageLimit(100. * geant::units::TeV);
      gsMSCModelOpt0->SetLowEnergyUsageLimit(100. * geant::units::eV);
      gsMSCModelOpt0->SetHighEnergyUsageLimit(100. * geant::units::TeV);
      // set opt1 to be inactivated in region #11 (HcalRegion)
      gsMSCModelOpt1->AddToUserRequestedInActiveRegions(11);
      // set opt0 to be inactivated in all the 0-21 regions but except in #11 (HcalRegion)
      std::vector<int> inactiveRegionIndexVect(22);
      std::iota(std::begin(inactiveRegionIndexVect), std::end(inactiveRegionIndexVect), 0);
      inactiveRegionIndexVect.erase(inactiveRegionIndexVect.begin() + 11);
      gsMSCModelOpt0->AddToUserRequestedInActiveRegions(inactiveRegionIndexVect);
      //
      // add the 2 MSC model to the MSC process
      eMSCProc->AddModel(gsMSCModelOpt1);
      eMSCProc->AddModel(gsMSCModelOpt0);
      // add process to particle
      AddProcessToParticle(particle, eMSCProc);
    }
    if (particle == geantphysics::Positron::Definition()) {
      // std::cout<<"  Positron" <<std::endl;
      //
      // create ionization process for e+ with 1 model:
      //
      geantphysics::EMPhysicsProcess *eIoniProc = new geantphysics::ElectronIonizationProcess("e+Ioni");
      // create the Moller-Bhabha model for ionization i.e. for e+ + e- -> e+ + e- intercation
      geantphysics::EMModel *eMBModel = new geantphysics::MollerBhabhaIonizationModel(false);
      eMBModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      eMBModel->SetLowEnergyUsageLimit(1.0 * geant::units::keV);
      eMBModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      eMBModel->SetUseSamplingTables(fUseSamplingTables);
      // add the model to the process
      eIoniProc->AddModel(eMBModel);
      // add the process to the e+ particle
      AddProcessToParticle(particle, eIoniProc);
      //
      // create bremsstrahlung process for e+ with 2 models:
      //
      geantphysics::EMPhysicsProcess *eBremProc = new geantphysics::ElectronBremsstrahlungProcess("e+Brem");
      // create a SeltzerBergerBremsModel for e-
      geantphysics::EMModel *eSBModel = new geantphysics::SeltzerBergerBremsModel(false);
      // set min/max energies of the model
      eSBModel->SetLowEnergyUsageLimit(1.0 * geant::units::keV);
      eSBModel->SetHighEnergyUsageLimit(1.0 * geant::units::GeV);
      eSBModel->SetUseSamplingTables(fUseSamplingTables);
      // how to inactivate this model in a given region i.e. region with index 1
      // active regions for a model are set based on their process active regions + user requested inactive regions
      // eSBModel->AddToUserRequestedInActiveRegions(1);
      //
      // add this model to the process
      eBremProc->AddModel(eSBModel);
      //
      // create a RelativisticBremsModel for e+
      geantphysics::EMModel *eRelBModel = new geantphysics::RelativisticBremsModel();
      eRelBModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      eRelBModel->SetLowEnergyUsageLimit(1.0 * geant::units::GeV);
      eRelBModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      eRelBModel->SetUseSamplingTables(fUseSamplingTables);
      // add this model to the process
      eBremProc->AddModel(eRelBModel);
      //
      // add the process to the e+ particle
      AddProcessToParticle(particle, eBremProc);
      //
      // create MSC process with 2 models:
      // = 1. with range-factor = 0.2  (EM-opt1) settings and active everywhere but in HcalRegion (#11)
      // = 2. with range-factor = 0.06 (EM-opt0) settings and active only in HcalRegion (#11)
      geantphysics::EMPhysicsProcess *eMSCProc = new geantphysics::MSCProcess("e+msc");
      // create GS-msc model, set min/max usage limits
      geantphysics::MSCModel *gsMSCModelOpt1 = nullptr;
      geantphysics::MSCModel *gsMSCModelOpt0 = nullptr;
      if (fVectorizedMSC) {
        gsMSCModelOpt1 = new geantphysics::GSMSCModelSimplified();
        gsMSCModelOpt0 = new geantphysics::GSMSCModelSimplified();
      } else {
        gsMSCModelOpt1 = new geantphysics::GSMSCModel();
        gsMSCModelOpt0 = new geantphysics::GSMSCModel();
      }
      gsMSCModelOpt1->SetBasketizable(fVectorizedMSC);
      gsMSCModelOpt0->SetBasketizable(fVectorizedMSC);
      // one with opt1 and one with opt0 stepping i.e. RangeFactor = 0.2 and 0.06
      gsMSCModelOpt1->SetRangeFactor(0.2);
      gsMSCModelOpt0->SetRangeFactor(0.06);
      //
      gsMSCModelOpt1->SetMSCSteppingAlgorithm(geantphysics::MSCSteppingAlgorithm::kUseSaftey);
      gsMSCModelOpt0->SetMSCSteppingAlgorithm(geantphysics::MSCSteppingAlgorithm::kUseSaftey);
      //
      gsMSCModelOpt1->SetLowEnergyUsageLimit(100. * geant::units::eV);
      gsMSCModelOpt1->SetHighEnergyUsageLimit(100. * geant::units::TeV);
      gsMSCModelOpt0->SetLowEnergyUsageLimit(100. * geant::units::eV);
      gsMSCModelOpt0->SetHighEnergyUsageLimit(100. * geant::units::TeV);
      // set opt1 to be inactivated in region #11 (HcalRegion)
      gsMSCModelOpt1->AddToUserRequestedInActiveRegions(11);
      // set opt0 to be inactivated in all the 0-21 regions but except in #11 (HcalRegion)
      std::vector<int> inactiveRegionIndexVect(22);
      std::iota(std::begin(inactiveRegionIndexVect), std::end(inactiveRegionIndexVect), 0);
      inactiveRegionIndexVect.erase(inactiveRegionIndexVect.begin() + 11);
      gsMSCModelOpt0->AddToUserRequestedInActiveRegions(inactiveRegionIndexVect);
      //
      // add the 2 MSC model to the MSC process
      eMSCProc->AddModel(gsMSCModelOpt1);
      eMSCProc->AddModel(gsMSCModelOpt0);
      // add process to particle
      AddProcessToParticle(particle, eMSCProc);
      //
      // create e+ electron annihilation into 2 gamma process
      geantphysics::PositronAnnihilationProcess *pAnhProc = new geantphysics::PositronAnnihilationProcess();
      AddProcessToParticle(particle, pAnhProc);
    }
    if (particle == geantphysics::Gamma::Definition()) {
      // create compton scattering process for gamma with 1 model:
      //
      geantphysics::EMPhysicsProcess *comptProc = new geantphysics::ComptonScatteringProcess();
      // create the Klein-Nishina model for Compton scattering i.e. for g + e- -> g + e- intercation
      geantphysics::EMModel *kncModel = new geantphysics::KleinNishinaComptonModel();
      kncModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      kncModel->SetLowEnergyUsageLimit(100.0 * geant::units::eV);
      kncModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      kncModel->SetUseSamplingTables(fUseSamplingTables);
      // add the model to the process
      comptProc->AddModel(kncModel);
      //
      // add the process to the gamma particle
      AddProcessToParticle(particle, comptProc);
      //
      // create gamma conversion process for gamma with 1 model:
      //
      geantphysics::EMPhysicsProcess *convProc = new geantphysics::GammaConversionProcess();
      // create the Bethe-Heitler model for pair production i.e. for g + A -> e- + e+ interaction
      geantphysics::EMModel *bhModel = new geantphysics::BetheHeitlerPairModel();
      bhModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      bhModel->SetLowEnergyUsageLimit(2.0 * geant::units::kElectronMassC2);
      bhModel->SetHighEnergyUsageLimit(80.0 * geant::units::GeV);
      bhModel->SetUseSamplingTables(fUseSamplingTables);
      // add the model to the process
      convProc->AddModel(bhModel);
      //
      // create the relativistic model(with LPM) for pair production i.e. for g + A -> e- + e+ interaction
      geantphysics::EMModel *relModel = new geantphysics::RelativisticPairModel();
      relModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      relModel->SetLowEnergyUsageLimit(80.0 * geant::units::GeV);
      relModel->SetHighEnergyUsageLimit(100.0 * geant::units::TeV);
      // turn off using sampling tables (no need cause it's used at very high >80 GeV energies)
      relModel->SetUseSamplingTables(false);
      // add the model to the process
      convProc->AddModel(relModel);
      //
      // add the process to the gamma particle
      AddProcessToParticle(particle, convProc);
      //
      // create photoelectric effect process for gamma with 1 model:
      //
      geantphysics::EMPhysicsProcess *photoelectricProc = new geantphysics::GammaPhotoElectricProcess();
      // create the Sauter-Gavrila model for photoelectric effect
      geantphysics::EMModel *sgModel = new geantphysics::SauterGavrilaPhotoElectricModel();
      sgModel->SetBasketizable(fVectorized);
      // set min/max energies of the model
      sgModel->SetLowEnergyUsageLimit(1.0 * geant::units::eV);
      sgModel->SetHighEnergyUsageLimit(1.0 * geant::units::TeV);
      sgModel->SetUseSamplingTables(fUseSamplingTables);
      // add the model to the process
      photoelectricProc->AddModel(sgModel);
      //
      // add the process to the gamma particle
      AddProcessToParticle(particle, photoelectricProc);
    }
  }
}

} // namespace cmsapp
