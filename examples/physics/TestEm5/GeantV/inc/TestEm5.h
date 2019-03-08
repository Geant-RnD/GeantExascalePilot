
#ifndef TESTEM5_H
#define TESTEM5_H

#ifndef GEANT_VAPPLICATION
#include "Geant/UserApplication.h"
#endif

#include "Geant/Typedefs.h"
#include "Geant/Fwd.h"
#include "Geant/TaskData.h"

namespace GEANT_IMPL_NAMESPACE {
namespace geant {
class RunManager;
class TaskDataHandle;
class Event;
class Track;
}
}

#include "Hist.h"
#include "TestEm5Data.h"

#include <mutex>
#include <vector>

namespace userapplication {

/**
 * @brief GeantV implementation of the Geant4 TestEm5 application.
 *
 * The application simulates passage of particles (type and energy configurable) through a simple thin target (material,
 * thickness, secondary production cuts are configurable). The main purpose of the simulation is to generate angular
 * distribution of charged particles transmitted through the target. However, several other quantities (like mean energy
 * deposit in target, energy leakage[both primary and secondary] and energy balance, mean number of charged and neutral
 * steps in the target, mean number of secondary partciles per particle type, transmitting/backscattering coefficients,
 * etc.) will also be collected and reported at the simulation.
 *
 * @class   TestEm5
 * @author  M Novak
 * @date    July 2017
 */

class TestEm5DetectorConstruction;
class TestEm5PrimaryGenerator;

/** @brief TestEm5 user application */
class TestEm5 : public geant::UserApplication {
public:
  /** @brief Constructor TestEm5 */
  TestEm5(geant::RunManager *runmgr, TestEm5DetectorConstruction *det, TestEm5PrimaryGenerator *gun);

  /** @brief Destructor TestEm5 */
  virtual ~TestEm5();

  /** @brief Interface method to allow registration of user defined thread local data. */
  virtual void AttachUserData(geant::TaskData *td);

  /** @brief Interface method to initialize the application. */
  virtual bool Initialize();

  /** @brief Interace method that is called at the end of each simulation step. */
  virtual void SteppingActions(geant::Track &track, geant::TaskData *td);

  /** @brief Interace method that is called when the transportation of an event (including all primary and their
    *        secondary particles) is completed .*/
  virtual void FinishEvent(geant::Event *event);

  /** @brief Interface method that is called at the end of the simulation (when the transportation of all events are
    *        are completed). */
  virtual void FinishRun();

  // Some application specific methods to set the angular distribution histogram parameters.
  void SetHist1FileName(const std::string &name) { fHist1FileName = name; }
  void SetHist1NumBins(int val) { fHist1NumBins = val; }
  void SetHist1Min(double val) { fHist1Min = val; }
  void SetHist1Max(double val) { fHist1Max = val; }

private:
  /** @brief Copy constructor TestEm5 (deleted) */
  TestEm5(const TestEm5 &) = delete;
  /** @brief Operator= for TestEm5 (deleted) */
  TestEm5 &operator=(const TestEm5 &) = delete;

private:
  std::string fHist1FileName;
  bool fInitialized;
  // ID of the target logical volume (used to check if the current step was done in the target)
  // this data will be obtained from the TestEm5DetectorConstruction at initialization
  int fTargetLogicalVolumeID;
  // some data regarding the number of primaries per event and number of buffered events (i.e. number of event-slots)
  // these data will be obtained from the RunManager::GeantConfig object at initialization
  int fNumPrimaryPerEvent;
  int fNumBufferedEvents;
  // histogram configuration data (can be changed from input arguments)
  int fHist1NumBins;
  double fHist1Min;
  double fHist1Max;
  //
  double fPrimaryParticleCharge;
  // user defined thread local data structure handlers to obtain the thread local data structures (defined and
  // registered by the user) during the simulation (in the SteppingActions(i.e. at the end of each simulation step),
  // Digitization(i.e. at the end of an event) and FinishRun (i.e. at the end of the simulation):
  //
  // 1. merged from all working threads when transportation of an event (with all corresponding primary and secondary
  //    particles) are completed
  geant::TaskDataHandle<TestEm5ThreadDataEvents> *fDataHandlerEvents;
  // 2. merged from all working threads when transportation of all events (i.e. end of the simulation) are completed
  geant::TaskDataHandle<TestEm5ThreadDataRun> *fDataHandlerRun;
  // a unique, run-global user defined data structure to store cumulated quantities per primary particle during the
  // simulation
  TestEm5Data *fData;
  //
  TestEm5DetectorConstruction *fDetector;
  TestEm5PrimaryGenerator *fPrimaryGun;
  // mutex to prevent multiple threads writing into the unique, run-global TestEm5Data object (in the Digitization after
  // the merge of the user defined per-event data distributed among the threads)
  std::mutex fMutex;
};

} // namespace userapplication

#endif // TESTEM5_H
