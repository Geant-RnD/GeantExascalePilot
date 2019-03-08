
#include "CMSFullApp.h"

#include "Geant/Event.h"
#include "Geant/RunManager.h"
#include "Geant/TaskData.h"

#include "Geant/Error.h"

#include "Geant/SystemOfUnits.h"

#include "Geant/Particle.h"

// just for the printout at the end
#include "CMSParticleGun.h"

#include "Geant/MaterialCuts.h"

#include <iostream>
#include <iomanip>

#include "Geant/Region.h"

namespace cmsapp {

CMSFullApp::CMSFullApp(geant::RunManager *runmgr, CMSParticleGun *gun) : geant::UserApplication(runmgr), fGun(gun)
{
  fIsPerformance      = false;
  fInitialized        = false;
  fNumPrimaryPerEvent = CMSParticleGun::GetMaxNumberOfPrimariesPerEvent();
  fNumBufferedEvents  = -1;
  fDataHandlerEvents  = nullptr;
  fData               = nullptr;
}

CMSFullApp::~CMSFullApp()
{
  if (fData) {
    delete fData;
  }
}

void CMSFullApp::AttachUserData(geant::TaskData *td)
{
  if (fIsPerformance) {
    return;
  }
  // Create application specific thread local data structure to collecet/handle thread local multiple per-event data
  // structure. Provide number of event-slots and number of primaries per event
  CMSThreadDataEvents *eventData = new CMSThreadDataEvents(fNumBufferedEvents, fNumPrimaryPerEvent);
  fDataHandlerEvents->AttachUserData(eventData, td);
}

bool CMSFullApp::Initialize()
{
  int numRegion = (vecgeom::Region::GetTheRegionTable()).size();
  std::vector<int> matCutPerregion(numRegion, 0);
  std::cout << (vecgeom::Region::GetTheRegionTable()).size();
  int sumNumMC                                             = 0;
  const std::vector<geantphysics::MaterialCuts *> &mcTable = geantphysics::MaterialCuts::GetTheMaterialCutsTable();
  for (size_t imc = 0; imc < mcTable.size(); ++imc)
    ++matCutPerregion[mcTable[imc]->GetRegionIndex()];

  for (int ir = 0; ir < numRegion; ++ir) {
    std::cout << " ir = " << ir << " region Name = " << (vecgeom::Region::GetTheRegionTable())[ir]->GetName()
              << " #mc = " << matCutPerregion[ir] << std::endl;
    sumNumMC += matCutPerregion[ir];
  }
  std::cout << " === Total number of MC = " << sumNumMC << " vs " << mcTable.size() << std::endl;

  if (fIsPerformance) {
    return true;
  }
  // Initialize application. Geometry must be loaded.
  if (fInitialized) return true;
  //
  // get number of primary per event and number of event-slots from geant::GeantConfig
  int maxPrimPerEvt = fGun->GetNumPrimaryPerEvt();
  // if it was set by the user
  if (maxPrimPerEvt > 0) {
    fNumPrimaryPerEvent = maxPrimPerEvt;
  }
  fNumBufferedEvents = fRunMgr->GetConfig()->fNbuff;
  //
  // register thread local user data and get handler for them
  fDataHandlerEvents = fRunMgr->GetTDManager()->RegisterUserData<CMSThreadDataEvents>("CMSAppThreadDataEvents");
  //
  // create the unique, global data struture that will be used to store cumulated per-primary data during the simulation
  fData = new CMSData(CMSParticleGun::GetNumberOfPrimaryTypes());
  //
  fInitialized = true;
  return true;
}

void CMSFullApp::SteppingActions(geant::Track &track, geant::TaskData *td)
{
  if (fIsPerformance) {
    return;
  }
  // get some particle properties
  const geantphysics::Particle *part = geantphysics::Particle::GetParticleByInternalCode(track.GVcode());
  int pdgCode                        = part->GetPDGCode();
  double charge                      = part->GetPDGCharge();
  //
  // get the user defined thread local data structure per-primary particle for: the event-slot index (that defines the
  // per-event data structure) and the primary index (that defines the per-primary data structure within that per-event
  // data structure). NOTE: each tracks stores the event-slot and primary partcile index that event and primary particle
  // within that event the track belongs to.
  CMSDataPerPrimary &dataPerPrimary =
      (*fDataHandlerEvents)(td).GetDataPerEvent(track.EventSlot()).GetDataPerPrimary(track.PrimaryParticleIndex());
  // do the scoring:
  // 1. collet charged/neutral steps that were done in the target (do not count the creation step i.e. secondary tracks
  //    that has just been added in this step)
  if (track.Status() != geant::kNew) {
    if (charge == 0.0) {
      dataPerPrimary.AddNeutralStep();
      dataPerPrimary.AddNeutralTrackL(track.GetStep());
    } else {
      dataPerPrimary.AddChargedStep();
      dataPerPrimary.AddChargedTrackL(track.GetStep());
    }
    dataPerPrimary.AddEdep(track.Edep());
  }
  // Get the MaterialCuts from the LogicalVolume
  const geantphysics::MaterialCuts *matCut = static_cast<const geantphysics::MaterialCuts *>(
      (const_cast<vecgeom::LogicalVolume *>(track.GetVolume())->GetMaterialCutsPtr()));
  int regIndex = matCut->GetRegionIndex();
  dataPerPrimary.AddStepPerRegion(regIndex);

  // collect secondary particle type statistics
  if (track.Status() == geant::kNew) {
    switch (pdgCode) {
    // gamma
    case 22:
      dataPerPrimary.AddGamma();
      break;
    // e
    case 11:
      dataPerPrimary.AddElectron();
      break;
    // e+
    case -11:
      dataPerPrimary.AddPositron();
      break;
    }
  }
}

void CMSFullApp::FinishEvent(geant::Event *event)
{
  if (fIsPerformance) {
    return;
  }
  // merge the thread local data (filled in the SteppingActions() and distributed now in the different threads) that
  // belongs to the event (that occupied a given event-slot) that has been just transported
  CMSThreadDataEvents *data = fRunMgr->GetTDManager()->MergeUserData(event->GetSlot(), *fDataHandlerEvents);
  // after the merge, we write the data into the user defined unique, global data structure. However, since more than
  // one thread can write into this global data structure, we need to protect the global data object by a lock:
  CMSDataPerEvent &dataPerEvent = data->GetDataPerEvent(event->GetSlot());

  fMutex.lock();
  // get the event number and print
  int nPrims = event->GetNprimaries();
  std::cout << " \n================================================================= \n"
            << " ===  FinishEvent  --- event = " << event->GetEvent() << " with " << nPrims << " primary:" << std::endl;
  for (int ip = 0; ip < nPrims; ++ip) {
    geant::Track *primTrack     = event->GetPrimary(ip);
    int primGVCode              = primTrack->GVcode();
    const std::string &primName = geantphysics::Particle::GetParticleByInternalCode(primGVCode)->GetName();
    int primTypeIndx            = CMSParticleGun::GetPrimaryTypeIndex(primName);
    double primEkin             = primTrack->Ekin();
    double xdir                 = primTrack->Dx();
    double ydir                 = primTrack->Dy();
    double zdir                 = primTrack->Dz();
    fData->AddDataPerPrimaryType(dataPerEvent.GetDataPerPrimary(ip), primTypeIndx);
    std::cout << "  Primary Particle:  " << ip << " (type inedx = " << primTypeIndx << ")\n"
              << "    Name      =  " << primName << " \n"
              << "    Energy    =  " << primEkin / geant::units::GeV << " [GeV]\n"
              << "    Direction = (" << xdir << ", " << ydir << ", " << zdir << ") \n";
    dataPerEvent.GetDataPerPrimary(ip).Print();
  }
  fMutex.unlock();
  // clear the currently added ("master") thread local data (the event-slot, where the currently finished event was)

  data->Clear(event->GetSlot());
  return;
}

void CMSFullApp::FinishRun()
{
  // print run conditions
  CMSParticleGun::Print();
  if (fIsPerformance) {
    return;
  }
  //
  //
  int numPrimTypes = fData->GetNumberOfPrimaryTypes();
  int numPrimaries = 0;
  int nTotSteps    = 0;
  int nReg         = fData->GetDataPerPrimaryType(0).GetNRegions();
  std::vector<int> nTotStepsPerReg(nReg, 0);
  for (int ipt = 0; ipt < numPrimTypes; ++ipt) {
    numPrimaries += fData->GetDataPerPrimaryType(ipt).GetNumPrimaries();
    for (size_t i = 0; i < fData->GetDataPerPrimaryType(ipt).GetNRegions(); ++i) {

      nTotStepsPerReg[i] += fData->GetDataPerPrimaryType(ipt).GetStepPerRegion(i);
      nTotSteps += fData->GetDataPerPrimaryType(ipt).GetStepPerRegion(i);
      // std::cout <<" FinishRun:: summing up steps per region: "<< i << ", adding: " <<
      // fData->GetDataPerPrimaryType(ipt).GetStepPerRegion(i) << std::endl;
    }
  }

  //
  std::ios::fmtflags mode = std::cout.flags();
  int prec                = std::cout.precision(2);
  std::cout << " \n ==================================   Run summary   ===================================== \n"
            << std::endl;
  std::cout << std::setprecision(4);
  //  std::cout<< "    Number of events        = " << numEvents                                                     <<
  //  std::endl;
  std::cout << "    Total number of primary = " << numPrimaries << std::endl;
  std::cout << " \n ---------------------------------------------------------------------------------------- \n"
            << std::endl;
  // compute and print run statistics per primary type per primary
  for (int ipt = 0; ipt < numPrimTypes; ++ipt) {
    const CMSDataPerPrimaryType &runData = fData->GetDataPerPrimaryType(ipt);
    int nPrimaries                       = runData.GetNumPrimaries();
    double norm                          = static_cast<double>(nPrimaries);
    if (norm > 0.) {
      norm = 1. / norm;
    } else {
      continue;
    }
    // compute and print statistic
    //
    std::string primName = CMSParticleGun::GetPrimaryName(ipt);
    double meanEdep      = runData.GetEdep() * norm;
    double rmsEdep       = std::sqrt(std::abs(runData.GetEdep2() * norm - meanEdep * meanEdep));
    double meanLCh       = runData.GetChargedTrackL() * norm;
    double rmsLCh        = std::sqrt(std::abs(runData.GetChargedTrackL2() * norm - meanLCh * meanLCh));
    double meanLNe       = runData.GetNeutralTrackL() * norm;
    double rmsLNe        = std::sqrt(std::abs(runData.GetNeutralTrackL2() * norm - meanLNe * meanLNe));
    double meanStpCh     = runData.GetChargedSteps() * norm;
    double rmsStpCh      = std::sqrt(std::abs(runData.GetChargedSteps2() * norm - meanStpCh * meanStpCh));
    double meanStpNe     = runData.GetNeutralSteps() * norm;
    double rmsStpNe      = std::sqrt(std::abs(runData.GetNeutralSteps2() * norm - meanStpNe * meanStpNe));
    double meanNGam      = runData.GetGammas() * norm;
    double rmsNGam       = std::sqrt(std::abs(runData.GetGammas2() * norm - meanNGam * meanNGam));
    double meanNElec     = runData.GetElectrons() * norm;
    double rmsNElec      = std::sqrt(std::abs(runData.GetElectrons2() * norm - meanNElec * meanNElec));
    double meanNPos      = runData.GetPositrons() * norm;
    double rmsNPos       = std::sqrt(std::abs(runData.GetPositrons2() * norm - meanNPos * meanNPos));

    std::cout << "  Number of primaries        = " << nPrimaries << "  " << primName << std::endl;
    std::cout << "  Total energy deposit per primary = " << meanEdep << " +- " << rmsEdep << " [GeV]" << std::endl;
    std::cout << std::endl;
    std::cout << "  Total track length (charged) per primary = " << meanLCh << " +- " << rmsLCh << " [cm]" << std::endl;
    std::cout << "  Total track length (neutral) per primary = " << meanLNe << " +- " << rmsLNe << " [cm]" << std::endl;
    std::cout << std::endl;
    std::cout << "  Number of steps (charged) per primary = " << meanStpCh << " +- " << rmsStpCh << std::endl;
    std::cout << "  Number of steps (neutral) per primary = " << meanStpNe << " +- " << rmsStpNe << std::endl;
    std::cout << std::endl;
    std::cout << "  Number of secondaries per primary : " << std::endl
              << "     Gammas    =  " << meanNGam << " +- " << rmsNGam << std::endl
              << "     Electrons =  " << meanNElec << " +- " << rmsNElec << std::endl
              << "     Positrons =  " << meanNPos << " +- " << rmsNPos << std::endl;
    std::cout << "  Total number of steps is : " << nTotSteps << std::endl;
    std::cout << "  Steps per region: " << std::endl;
    int numRegion = (vecgeom::Region::GetTheRegionTable()).size();
    for (int i = 0; i < numRegion; i++)
      std::cout << "     " << i << ": " << nTotStepsPerReg[i] << "\t\t "
                << 100 * (double)nTotStepsPerReg[i] / (double)nTotSteps << " % of the total." << std::endl;
    std::cout << " ......................................................................................... \n"
              << std::endl;
  }
  std::cout << " \n ======================================================================================== \n"
            << std::endl;

  std::cout.setf(mode, std::ios::floatfield);
  std::cout.precision(prec);
}
} // namespace cmsapp
