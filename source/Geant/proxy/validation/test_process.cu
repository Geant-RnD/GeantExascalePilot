#include "Geant/proxy/ProxyRunManager.hpp"
#include "Geant/proxy/ProxyHepEvtGenerator.hpp"
#include "Geant/proxy/ProxyEventServer.hpp"
#include "Geant/proxy/ProxyDeviceManager.hpp"

#ifndef ADDED_SUPPORT_FOR_SEPARABLE_COMPILATION
#include "Geant/proxy/src/ProxyRunManager.cpp"
#include "Geant/proxy/src/ProxyHepEvtGenerator.cpp"
#include "Geant/proxy/src/ProxyEventServer.cpp"
#include "Geant/proxy/src/ProxyDeviceManager.cpp"
#endif

using namespace geantx;

int main(int argc, char* argv[]) {

  //configuration - arguments

  const int number_of_events_to_process = 50;
  const char* event_filename = "hepevt.data";

  // run manager - register generator, detector, physics list and user actions

  ProxyRunManager<ProxyHepEvtGenerator> *runMgr 
    = ProxyRunManager<ProxyHepEvtGenerator>::Instance();

  ProxyHepEvtGenerator* agen = 
    new ProxyHepEvtGenerator(number_of_events_to_process,
                             event_filename);

  runMgr->RegisterEventGenerator(agen);
  
  // test
  runMgr->GetEventManager()->GenerateEvents(50);

  runMgr->GetDeviceManager()->SetPerformanceFlag(true);

  runMgr->GetDeviceManager()->UploadTrackData();

  runMgr->GetDeviceManager()->DoStep();

  runMgr->GetDeviceManager()->DownloadTrackData();

  runMgr->GetDeviceManager()->DeallocateTrackData();

  return 0;
}
