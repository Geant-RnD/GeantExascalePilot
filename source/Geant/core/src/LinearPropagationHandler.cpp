#include "Geant/core/LinearPropagationHandler.hpp"

#include "Geant/geometry/NavigationInterface.hpp"
#include "Geant/track/TrackState.hpp"

#include "Geant/core/Logger.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool LinearPropagationHandler::Propagate(TrackState &track, TaskData *td)
{
  // Scalar geometry length computation. The track is moved into the output basket.

  // Do straight propagation to physics process or boundary
  // if (track->GetSnext() < 1.E-8) td->fNsmall++;
  LinearStep(track, track.fGeometryState.fSnext);
  // Update total number of steps
  // td->fNsteps++;
  int nsmall = 0;

  bool status = true;
  if (track.fGeometryState.fBoundary) {
    track.fStatus = TrackStatus::Boundary;
    // Find out location after boundary
    while (IsSameLocation(track, td)) {
      nsmall++;
      if (nsmall > 10) {
        // Most likely a nasty overlap, some smarter action required. For now, just
        // kill the track.

        Log(kError) << "LinearPropagator: track " << track.fHistoryState.fParticle
                    << " from event " << track.fHistoryState.fEvent
                    << " stuck -> killing it";
        track.fStatus = TrackStatus::Killed;
        // Deposit track energy, then go directly to stepping actions
        // track->Stop();
        track.fPhysicsState.fEkin = 0.0;
        // jump to track->SetStage(kSteppingActionsStage);
        // record number of killed particles: td->fNkilled++;
        status = false;
        break;
      }
      LinearStep(track, 1.E-3);
    }
  } else {
    track.fStatus = TrackStatus::Physics;
    // Update number of steps to physics
    // td->fNphys++;
  }

  if (track.fGeometryState.fSnext < 1.E-8) track.fGeometryState.fSnext = 0.0;
  if (track.fGeometryState.fSafety < 1.E-8) track.fGeometryState.fSafety = 0.0;

  // Update time of flight and number of interaction lengths
  //  track->Time += track->TimeStep(track->fStep);
  //  track->fNintLen -= track->fStep/track->fIntLen;

  return status;
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool LinearPropagationHandler::IsSameLocation(TrackState &track, TaskData *td) const
{
  // Query geometry if the location has changed for a track
  if (track.fGeometryState.fSafety > 1.E-10 && track.fGeometryState.fSnext > 1.E-10) {
    // Track stays in the same volume
    track.fGeometryState.fBoundary = false;
    return true;
  }

  // It might be advantageous to not create the state each time.
  // vecgeom::NavigationState *tmpstate = td->GetPath();
  vecgeom::NavigationState *tmpstate =
      vecgeom::NavigationState::MakeInstance(track.fGeometryState.fPath->GetMaxLevel());

  bool same = NavigationInterface::IsSameLocation(track, *tmpstate);
  if (same) return true;
  if (track.fGeometryState.fNextpath->IsOutside())
    track.fStatus = TrackStatus::ExitingSetup;
  return false;
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
