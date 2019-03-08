
#ifndef CMSDATA_H
#define CMSDATA_H

#include <vector>
#include "Geant/Region.h"

namespace cmsapp {

//--------------------------------------------------------------------------------------------------------------------//
class CMSDataPerPrimary {
public:
  CMSDataPerPrimary();
  ~CMSDataPerPrimary()
  { /*nothing to do*/
  }

  void AddChargedStep() { fNumChargedSteps += 1.; }
  double GetChargedSteps() const { return fNumChargedSteps; }

  void AddNeutralStep() { fNumNeutralSteps += 1.; }
  double GetNeutralSteps() const { return fNumNeutralSteps; }

  void AddChargedTrackL(double val) { fChargedTrackL += val; }
  double GetChargedTrackL() const { return fChargedTrackL; }

  void AddNeutralTrackL(double val) { fNeutralTrackL += val; }
  double GetNeutralTrackL() const { return fNeutralTrackL; }

  void AddGamma() { fNumGammas += 1.; }
  double GetGammas() const { return fNumGammas; }

  void AddElectron() { fNumElectrons += 1.; }
  double GetElectrons() const { return fNumElectrons; }

  void AddPositron() { fNumPositrons += 1.; }
  double GetPositrons() const { return fNumPositrons; }

  void AddEdep(double val) { fEdep += val; }
  double GetEdep() const { return fEdep; }

  void AddStepPerRegion(int i) { fNumStepPerRegion[i] += 1.; }
  double GetStepsPerRegion(int i) const { return fNumStepPerRegion[i]; }
  size_t GetNumOfRegions() const { return fNumStepPerRegion.size(); }

  void Clear();
  CMSDataPerPrimary &operator+=(const CMSDataPerPrimary &other);

  void Print();

private:
  double fNumChargedSteps;            // mean number of charged steps per primary
  double fNumNeutralSteps;            // mean number of neutral steps per primary
  double fChargedTrackL;              // mean number of charged track length per primary
  double fNeutralTrackL;              // mean number of neutral track length per primary
  double fNumGammas;                  // mean number of secondary gamma particles per primary
  double fNumElectrons;               // mean number of secondary electron particles per primary
  double fNumPositrons;               // mean number of secondary positron particles per primary
  double fEdep;                       // mean energy deposit per primary
  std::vector<int> fNumStepPerRegion; // number of steps per region
};

//--------------------------------------------------------------------------------------------------------------------//
// Data structure to accumulate per-primary data for a given primary type during the simulation. Used in the global
// CMData object and there will be as many object from this class as many different primary particle candidates
class CMSDataPerPrimaryType {
public:
  CMSDataPerPrimaryType();
  ~CMSDataPerPrimaryType()
  { /*nothing to do*/
  }

  void AddOnePrimary() { fNumPrimaries += 1.; }
  double GetNumPrimaries() const { return fNumPrimaries; }

  void AddChargedSteps(double val)
  {
    fNumChargedSteps += val;
    fNumChargedSteps2 += val * val;
  }
  double GetChargedSteps() const { return fNumChargedSteps; }
  double GetChargedSteps2() const { return fNumChargedSteps2; }

  void AddNeutralSteps(double val)
  {
    fNumNeutralSteps += val;
    fNumNeutralSteps2 += val * val;
  }
  double GetNeutralSteps() const { return fNumNeutralSteps; }
  double GetNeutralSteps2() const { return fNumNeutralSteps2; }

  void AddChargedTrackL(double val)
  {
    fChargedTrackL += val;
    fChargedTrackL2 += val * val;
  }
  double GetChargedTrackL() const { return fChargedTrackL; }
  double GetChargedTrackL2() const { return fChargedTrackL2; }

  void AddNeutralTrackL(double val)
  {
    fNeutralTrackL += val;
    fNeutralTrackL2 += val * val;
  }
  double GetNeutralTrackL() const { return fNeutralTrackL; }
  double GetNeutralTrackL2() const { return fNeutralTrackL2; }

  void AddGammas(double val)
  {
    fNumGammas += val;
    fNumGammas2 += val * val;
  }
  double GetGammas() const { return fNumGammas; }
  double GetGammas2() const { return fNumGammas2; }

  void AddElectrons(double val)
  {
    fNumElectrons += val;
    fNumElectrons2 += val * val;
  }
  double GetElectrons() const { return fNumElectrons; }
  double GetElectrons2() const { return fNumElectrons2; }

  void AddPositrons(double val)
  {
    fNumPositrons += val;
    fNumPositrons2 += val * val;
  }
  double GetPositrons() const { return fNumPositrons; }
  double GetPositrons2() const { return fNumPositrons2; }

  void AddEdep(double val)
  {
    fEdep += val;
    fEdep2 += val * val;
  }

  int GetStepPerRegion(int i) const { return fNumStepPerRegion[i]; }
  size_t GetNRegions() const { return fNumStepPerRegion.size(); }
  void AddStepPerRegion(int ir, int nsteps) { fNumStepPerRegion[ir] += nsteps; }
  double GetEdep() const { return fEdep; }
  double GetEdep2() const { return fEdep2; }

  void Clear();
  void AddDataPerPrimary(CMSDataPerPrimary &data);

private:
  double fNumPrimaries;     // number of primaries from this type
  double fNumChargedSteps;  // mean number of charged steps per primary
  double fNumChargedSteps2; // mean number of charged steps per primary square
  double fNumNeutralSteps;  // mean number of neutral steps per primary
  double fNumNeutralSteps2; // mean number of neutral steps per primary square

  double fChargedTrackL;  // mean number of charged track length per primary
  double fChargedTrackL2; // mean number of charged track length  per primary square
  double fNeutralTrackL;  // mean number of neutral track length  per primary
  double fNeutralTrackL2; // mean number of neutral track length  per primary square

  double fNumGammas;     // mean number of secondary gamma particles per primary
  double fNumGammas2;    // mean number of secondary gamma particles per primary square
  double fNumElectrons;  // mean number of secondary electron particles per primary
  double fNumElectrons2; // mean number of secondary electron particles per primary square
  double fNumPositrons;  // mean number of secondary positron particles per primary
  double fNumPositrons2; // mean number of secondary positron particles per primary square

  double fEdep;                       // mean energy deposit per primary in the target
  double fEdep2;                      // mean energy deposit per primary in the target square
  std::vector<int> fNumStepPerRegion; // number of steps per region
};

//--------------------------------------------------------------------------------------------------------------------//
// Global data structure to accumulate per-primary data for all primary types during the simulation.
class CMSData {
public:
  CMSData(int nprimtypes = 1);
  ~CMSData();

  void Clear();
  // add data after one primary particle finished tracking with a given primary type index
  void AddDataPerPrimaryType(CMSDataPerPrimary &data, int ptypeindx);
  // for the end-of-run
  const CMSDataPerPrimaryType &GetDataPerPrimaryType(const int ptypeindx) const
  {
    return fDataPerPrimaryType[ptypeindx];
  }

  int GetNumberOfPrimaryTypes() const { return fNumPrimaryTypes; }

private:
  int fNumPrimaryTypes;
  std::vector<CMSDataPerPrimaryType> fDataPerPrimaryType;
};

//--------------------------------------------------------------------------------------------------------------------//
// Data structure per-event for (contains as many per-primary data structures as number of primaries in the event)
class CMSDataPerEvent {
public:
  CMSDataPerEvent(int nprimperevent = 1);
  ~CMSDataPerEvent()
  { /*nothing to do*/
  }

  int GetNumberOfPrimaryPerEvent() const { return fNumPrimaryPerEvent; }
  void Clear();
  CMSDataPerPrimary &GetDataPerPrimary(int primindx) { return fPerPrimaryData[primindx]; }
  CMSDataPerEvent &operator+=(const CMSDataPerEvent &other);

private:
  int fNumPrimaryPerEvent;
  std::vector<CMSDataPerPrimary> fPerPrimaryData; // as many as primary particle in an event
};

//--------------------------------------------------------------------------------------------------------------------//
// Thread local data structure for CMSapp to collect/handle thread local multiple per-event data structures (as many
// per-event data structures as number of events are transported on the same time). Each of the currently transported
// events occupies one possible event-slot and per-event data can be identified by the index of this event-slot. This
// user defined thread local data needs to implement both the Merge and Clear methods for a given event-slot index:
// these methods are called when a data per-event,that corresponds to the completed event(with a given event-slot
// index),
// is merged from all threads.
class CMSThreadDataEvents {
public:
  CMSThreadDataEvents(int nevtbuffered, int nprimperevent);
  ~CMSThreadDataEvents()
  { /*nothing to do*/
  }

  void Clear(int evtslotindx) { fPerEventData[evtslotindx].Clear(); }
  bool Merge(int evtslotindx, const CMSThreadDataEvents &other);

  CMSDataPerEvent &GetDataPerEvent(int evtslotindx) { return fPerEventData[evtslotindx]; }
  const CMSDataPerEvent &GetDataPerEvent(int evtslotindx) const { return fPerEventData[evtslotindx]; }

private:
  int fNumBufferedEvents;
  std::vector<CMSDataPerEvent> fPerEventData;
};
} // namespace cmsapp

#endif // CMSDATA_H
