
#include "LHCbFullApp.h"

#include "Geant/Event.h"
#include "Geant/RunManager.h"
#include "Geant/TaskData.h"

#include "Geant/FactoryStore.h"

#include "Geant/Error.h"

#include "Geant/SystemOfUnits.h"

#include "Geant/Particle.h"

// just for the printout at the end
#include "LHCbParticleGun.h"

#include <iostream>
#include <iomanip>

namespace lhcbapp {

LHCbFullApp::LHCbFullApp(geant::RunManager *runmgr, LHCbParticleGun *gun) : geant::UserApplication(runmgr), fGun(gun)
{
  fIsPerformance      = false;
  fInitialized        = false;
  fNumPrimaryPerEvent = LHCbParticleGun::GetMaxNumberOfPrimariesPerEvent();
  fNumBufferedEvents  = -1;
  fDataHandlerEvents  = nullptr;
  fData               = nullptr;

#ifdef USE_ROOT
  FactoryStore *store = FactoryStore::Instance(runmgr->GetConfig()->fNbuff);
  fFactory            = store->GetFactory<MyHit>(16, runmgr->GetNthreadsTotal());
  // set factory to use thread-local queues
  fFactory->queue_per_thread = true;
#endif
}

LHCbFullApp::~LHCbFullApp()
{
  delete fData;
#ifdef USE_ROOT
  delete fMerger;
#endif
}

void LHCbFullApp::AttachUserData(geant::TaskData *td)
{
  if (fIsPerformance) {
    return;
  }

  // Create application specific thread local data structure to collecet/handle thread local multiple per-event data
  // structure. Provide number of event-slots and number of primaries per event
  LHCbThreadDataEvents *eventData = new LHCbThreadDataEvents(fNumBufferedEvents, fNumPrimaryPerEvent);

#ifdef USE_ROOT
  // create here the TTree
  eventData->fHitsFile = fMerger->GetFile();

  eventData->fHitsTree = new TTree("Tree", "Simulation output");
  eventData->fHitsTree->ResetBit(kMustCleanup);
  eventData->fHitsTree->Branch("hitblockoutput", "GeantBlock<MyHit>", &(eventData->fHitsBlock));
#endif

  fDataHandlerEvents->AttachUserData(eventData, td);
}

void LHCbFullApp::DeleteUserData(geant::TaskData *td)
{
  if (fIsPerformance) {
    return;
  }
  LHCbThreadDataEvents *eventData = fDataHandlerEvents->GetUserData(td);

#ifdef USE_ROOT
  // make sure that everything has been sent for merging
  //  std::cout << "Writing before delete" << std::endl;
  eventData->fHitsFile->Write();
#endif

  delete eventData;
}

bool LHCbFullApp::Initialize()
{
  if (fIsPerformance) {
    return true;
  }

#ifdef USE_ROOT
  // IO
  fMerger           = new ROOT::Experimental::TBufferMerger("simu.root");
  fOutputBlockWrite = 1000;
#endif

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
  fDataHandlerEvents = fRunMgr->GetTDManager()->RegisterUserData<LHCbThreadDataEvents>("LHCbAppThreadDataEvents");
  //
  // create the unique, global data struture that will be used to store cumulated per-primary data during the simulation
  fData = new LHCbData(LHCbParticleGun::GetNumberOfPrimaryTypes());
  //

  // Loop unique volume id's
  int nvolumes                         = fRunMgr->GetNvolumes();
  vector_t<Volume_t const *> &lvolumes = fRunMgr->GetVolumes();
  geant::Printf("Found %d logical volumes", nvolumes);
  const Volume_t *vol;
  std::string svol, smat;
  int nvelo = 0;
  int necal = 0;
  int nhcal = 0;
  for (int ivol = 0; ivol < nvolumes; ++ivol) {
    vol = lvolumes[ivol];
    if (!vol) break;
    int idvol = vol->id();
    svol      = vol->GetName();

    // VELO & TT
    if (svol.find("/dd/Geometry/BeforeMagnetRegion/Velo/Sensors/lvVeloRDetectorPU") == 0 ||
        svol.find("/dd/Geometry/BeforeMagnetRegion/Velo/Sensors/lvVeloPhiDetector") == 0 ||
        svol.find("/dd/Geometry/BeforeMagnetRegion/Velo/Sensors/lvVeloRDetector") == 0 ||
        svol.find("/dd/Geometry/BeforeMagnetRegion/TT/Modules/lvSensor") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/IT/Ladder/lvLong") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/IT/Ladder/lvShort") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/OT/Modules/lvS3ModuleA") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/OT/Modules/lvS1ModuleA") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/OT/Modules/lvLModuleA") == 0 ||
        svol.find("/dd/Geometry/AfterMagnetRegion/T/OT/Modules/lvS2ModuleA") == 0) {
      fSensFlags[idvol] = true;
      fVELOMap[idvol]   = nvelo;
      fVELOid[necal]    = idvol;
      nvelo++;
    }

    // ECAL cells
    if (svol.find("/dd/Geometry/DownstreamRegion/Ecal/Modules/InnCell") == 0 ||
        svol.find("/dd/Geometry/DownstreamRegion/Ecal/Modules/MidCell") == 0 ||
        svol.find("/dd/Geometry/DownstreamRegion/Ecal/Modules/OutCell") == 0) {
      fSensFlags[idvol] = true;
      fECALMap[idvol]   = necal;
      fECALid[necal]    = idvol;
      necal++;
    }

    // HCAL cells
    if (svol.find("/dd/Geometry/DownstreamRegion/Hcal/Cells/lvHcalInnCellUpScTile") == 0 ||
        svol.find("/dd/Geometry/DownstreamRegion/Hcal/Cells/lvHcalInnCellLowScTile") == 0 ||
        svol.find("/dd/Geometry/DownstreamRegion/Hcal/Cells/lvHcalOutCellScTile") == 0) {
      fSensFlags[idvol] = true;
      fHCALMap[idvol]   = nhcal;
      fHCALid[nhcal]    = idvol;
      nhcal++;
    }
  }

  geant::Printf("=== LHCbApplication::Initialize: necal=%d  nhcal=%d", necal, nhcal);

  fInitialized = true;
  return true;
}

void LHCbFullApp::SteppingActions(geant::Track &track, geant::TaskData *td)
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
  LHCbDataPerPrimary &dataPerPrimary =
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

  int tid = td->fTid;

  int ivol;
  int idtype;
  int mod;
  Volume_t const *vol;

  vol  = track.GetVolume();
  ivol = vol->id();

  idtype = 0;
  if (fSensFlags[ivol]) {

    if (vol->GetName()[30] == 'E')
      idtype = 1;
    else if (vol->GetName()[30] == 'H')
      idtype = 2;

    switch (idtype) {
    case 0:
      mod = fVELOMap.find(ivol)->second;
      fEdepVELO[mod][tid] += track.Edep();
      break;
    case 1:
      mod = fECALMap.find(ivol)->second;
      fEdepECAL[mod][tid] += track.Edep();
      break;
    case 2:
      mod = fHCALMap.find(ivol)->second;
      fEdepHCAL[mod][tid] += track.Edep();
      break;
    }

#ifdef USE_ROOT
    MyHit *hit;
    // Deposit hits
    //      if (idtype==1) {

    //   Printf("hit at z %f id %i", track.Z(), idtype);

    hit         = fFactory->NextFree(track.EventSlot(), tid);
    hit->fX     = track.X();
    hit->fY     = track.Y();
    hit->fZ     = track.Z();
    hit->fEdep  = 1000 * track.Edep();
    hit->fTime  = track.Time();
    hit->fEvent = track.Event();
    hit->fTrack = track.Particle();
    hit->fVolId = ivol;
    hit->fDetId = idtype;
#endif // USE_ROOT
  }

#ifdef USE_ROOT
  LHCbThreadDataEvents &tde = (*fDataHandlerEvents)(td);

  while (!(fFactory->fOutputsArray[tid].empty())) {
    tde.fHitsBlock = fFactory->fOutputsArray[tid].back();

    tde.fHitsTree->Fill();

    fFactory->fOutputsArray[tid].pop_back();

    // now we can recycle data memory
    fFactory->Recycle(tde.fHitsBlock, tid);
  }

  if (tde.fHitsTree->GetEntries() > fOutputBlockWrite) {
    //      std::cout << "Writing " << tde.tree->GetEntries()<< std::endl;
    tde.fHitsFile->Write();
  }
#endif // USE_ROOT
}

void LHCbFullApp::FinishEvent(geant::Event *event)
{
  if (fIsPerformance) {
    return;
  }
  // merge the thread local data (filled in the SteppingActions() and distributed now in the different threads) that
  // belongs to the event (that occupied a given event-slot) that has been just transported
  LHCbThreadDataEvents *data = fRunMgr->GetTDManager()->MergeUserData(event->GetSlot(), *fDataHandlerEvents);

  LHCbDataPerEvent &dataPerEvent = data->GetDataPerEvent(event->GetSlot());

  fMutex.lock();
  // get the event number and print
  int nPrims = event->GetNprimaries();
  std::cout << " \n================================================================= \n"
            << " ===  FinishEvent  --- event = " << event->GetEvent() << " with " << nPrims << " primary:" << std::endl;

  for (int ip = 0; ip < nPrims; ++ip) {
    geant::Track *primTrack     = event->GetPrimary(ip);
    int primGVCode              = primTrack->GVcode();
    const std::string &primName = geantphysics::Particle::GetParticleByInternalCode(primGVCode)->GetName();
    int primTypeIndx            = LHCbParticleGun::GetPrimaryTypeIndex(primName);

    fData->AddDataPerPrimaryType(dataPerEvent.GetDataPerPrimary(ip), primTypeIndx);
    /*
    double      primEkin         = primTrack->T();
    double      xdir             = primTrack->Dx();
    double      ydir             = primTrack->Dy();
    double      zdir             = primTrack->Dz();

    std::cout << "  Primary Particle:  " << ip  << " (type inedx = " << primTypeIndx  << ")\n"
              << "    Name      =  "     << primName                                  << " \n"
              << "    Energy    =  "     << primEkin/geant::units::GeV                       << " [GeV]\n"
              << "    Direction = ("     << xdir << ", " << ydir << ", " << zdir      << ") \n";

    dataPerEvent.GetDataPerPrimary(ip).Print();
    */
  }

  fMutex.unlock();
  // clear the currently added ("master") thread local data (the event-slot, where the currently finished event was)

  data->Clear(event->GetSlot());
  return;
}

void LHCbFullApp::FinishRun()
{
  // print run conditions
  LHCbParticleGun::Print();
  if (fIsPerformance) {
    return;
  }

  //
  //
  int numPrimTypes = fData->GetNumberOfPrimaryTypes();
  int numPrimaries = 0;
  for (int ipt = 0; ipt < numPrimTypes; ++ipt) {
    numPrimaries += fData->GetDataPerPrimaryType(ipt).GetNumPrimaries();
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
    const LHCbDataPerPrimaryType &runData = fData->GetDataPerPrimaryType(ipt);
    int nPrimaries                        = runData.GetNumPrimaries();
    double norm                           = static_cast<double>(nPrimaries);
    if (norm > 0.) {
      norm = 1. / norm;
    } else {
      continue;
    }
    // compute and print statistic
    //
    std::string primName = LHCbParticleGun::GetPrimaryName(ipt);
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
    std::cout << " ......................................................................................... \n"
              << std::endl;
  }
  std::cout << " \n ======================================================================================== \n"
            << std::endl;

  std::cout.setf(mode, std::ios::floatfield);
  std::cout.precision(prec);
}
}
