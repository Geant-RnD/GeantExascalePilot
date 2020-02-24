
#include "Geant/geometry/ScalarNavInterfaceVG.hpp"
#include "Geant/core/math_wrappers.hpp"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/VNavigator.h"

namespace geantx {

inline namespace GEANT_IMPL_NAMESPACE {

using namespace VECGEOM_NAMESPACE;

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void ScalarNavInterfaceVG::NavFindNextBoundaryAndStep(int ntracks, const double *pstep, const double *x,
                                                      const double *y, const double *z, const double *dirx,
                                                      const double *diry, const double *dirz,
                                                      const VolumePath_t **instate, VolumePath_t **outstate,
                                                      double *step, double *safe, bool *isonbdr)
{
  // Find the next boundary and state after propagating to the boundary.
  // Input:  ntracks - number of tracks
  //         pstep - proposed step
  //         x, y, z, dirx, diry, dirz - particle position and direction
  //         instate - input particle navigation state
  //         safe - estimated safety value for the input point
  //         isonbdr - starting point is on a boundary
  // Output: outstate - navigation state after propagation. If boundary further
  //           than proposed step, outstate has to match instate
  //         step - propagation step for which the state is sampled
  //         safety - calculated safety value for the input point
  //         isonbdr - propagated point is on a boundary

  typedef Vector3D<Precision> Vector3D_t;
  constexpr double gTolerance = 1.e-9;

  for (int itr = 0; itr < ntracks; ++itr) {
    // Check if current safety allows for the proposed step
    if (safe[itr] > pstep[itr]) {
      step[itr]      = pstep[itr];
      isonbdr[itr]   = false;
      *outstate[itr] = *instate[itr];
      continue;
    }
    VNavigator const *nav = instate[itr]->Top()->GetLogicalVolume()->GetNavigator();
    step[itr]             = nav->ComputeStepAndSafetyAndPropagatedState(
        Vector3D_t(x[itr], y[itr], z[itr]), Vector3D_t(dirx[itr], diry[itr], dirz[itr]),
        Math::Min<double>(1.E20, pstep[itr]), *instate[itr], *outstate[itr] /* the paths */, !isonbdr[itr], safe[itr]);
    step[itr] = Math::Max<double>(2. * gTolerance, step[itr] + 2. * gTolerance);
    safe[itr] = Math::Max<double>(safe[itr], 0);
    // onboundary with respect to new point
    isonbdr[itr] = outstate[itr]->IsOnBoundary();
    //#### To add small step detection and correction - see ScalarNavInterfaceTGeo ####//
  }
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void ScalarNavInterfaceVG::NavFindNextBoundaryAndStep(Track &track)
{
  constexpr double gTolerance = 1.e-9;
  std::cout<<" *** NavFindNxtBndry: track #"<< track.fHistoryState.fParticle
	   <<", mother(prim:gen): "<< track.fHistoryState.fMother
	   <<" ("<< track.fHistoryState.fPrimaryIndx <<":"<< track.fHistoryState.fGeneration <<")"
	   <<", Nsteps="<< track.fHistoryState.fNsteps
	   <<"\n";

  std::cout<<"  trkPhys: Ekin="<< track.fPhysicsState.fEkin
	   <<", Momentum="<< track.fPhysicsState.fMomentum
	   <<", Edep="<< track.fPhysicsState.fEdep
	   <<", fPstep="<< track.fPhysicsState.fPstep
	   <<", fProc="<< track.fPhysicsState.fProcess
	   <<"\n";

  auto *nextvol = track.fGeometryState.fNextpath->Top();
  std::cout<<"  trkGeom: vol="<< track.fGeometryState.fVolume
	   <<", fPath.top="<< track.fGeometryState.fPath->Top()->GetLabel()
	   <<", fNext.top="<< (nextvol ? nextvol->GetLabel() : "NULL")
	   <<", fSafety="<<  track.fGeometryState.fSafety
	   <<", fSnext="<<  track.fGeometryState.fSnext
	   <<", isOnBndry="<<  track.fGeometryState.fIsOnBoundaryPreStp
	   <<", bndry="<<  track.fGeometryState.fBoundary
	   <<", pending="<<  track.fGeometryState.fPending
	   <<"\n";

  // Check if current safety allows for the proposed step
  if (track.fGeometryState.fSafety > track.fPhysicsState.fPstep) {
    track.fStep = track.fPhysicsState.fPstep;
    track.fGeometryState.fBoundary = false;
    *track.fGeometryState.fNextpath = *track.fGeometryState.fPath;
    std::cout<<" Not a geom-mandated step, returning...\n";
    return;
  }

  VNavigator const *nav = track.fGeometryState.fPath -> Top()->GetLogicalVolume()->GetNavigator();
  double step           = track.fStep;
  nav->FindNextBoundaryAndStep(track.fPos, track.fDir, *track.fGeometryState.fPath, *track.fGeometryState.fNextpath,
			       Math::Min<double>(1.E20, track.fPhysicsState.fPstep), step);
  std::cout<<" Step? in="<< track.fStep <<", out="<< step <<"\n";
  double safety = track.fGeometryState.fSafety;
  double snext  = nav->ComputeStepAndSafetyAndPropagatedState(track.fPos, track.fDir,
							      Math::Min<double>(1.E20, track.fPhysicsState.fPstep),
							      *track.fGeometryState.fPath, *track.fGeometryState.fNextpath,
							      !track.fGeometryState.fBoundary, safety);

  snext = Math::Max<double>(2. * gTolerance, snext + 2. * gTolerance);
  std::cout<<" Snext? in="<< track.fGeometryState.fSnext <<", out="<< snext <<"\n";
  track.fGeometryState.fSnext = snext;
  track.fGeometryState.fSafety = Math::Max<double>(safety, 0);

  // on-boundary with respect to new point
  std::cout<<" Boundary: in="<< track.fGeometryState.fBoundary <<", out="<< track.fGeometryState.fNextpath->IsOnBoundary() <<"\n";
  track.fGeometryState.fBoundary = track.fGeometryState.fNextpath->IsOnBoundary();
  //#### To add small step detection and correction - see ScalarNavInterfaceTGeo ####//
}

//______________________________________________________________________________
void ScalarNavInterfaceVG::NavIsSameLocation(int ntracks, const double *x, const double *y, const double *z,
                                             const double * /*dirx*/, const double * /*diry*/, const double * /*dirz*/,
                                             const VolumePath_t **start, VolumePath_t **end, bool *same,
                                             VolumePath_t *tmpstate)
{
  //
  // Checks if the navigation states corresponding to positions (x,y,z) are the
  // same as the ones pointed by start. Update new states in end.
  // Input:  ntracks - number of tracks to be checked
  //         x,y,z   - arrays of positions
  //         start   - starting navigation paths to compare with
  // Output: end     - navigation paths corresponding to given positions
  //         same    - flags showing if the end and start positions are matching

  for (int itr = 0; itr < ntracks; ++itr) {
    // TODO: not using the direction yet here !!
    bool samepath = GlobalLocator::HasSamePath(Vector3D<double>(x[itr], y[itr], z[itr]), *start[itr], *tmpstate);
    if (!samepath) {
      tmpstate->CopyTo(end[itr]);
#ifdef VECGEOM_CACHED_TRANS
      end[itr]->UpdateTopMatrix();
#endif
    }
    same[itr] = samepath;
  }
}

//______________________________________________________________________________
void ScalarNavInterfaceVG::NavIsSameLocation(Track &track, bool &same, VolumePath_t *tmpstate)
{
  // TODO: not using the direction yet here !!
  bool samepath =
      GlobalLocator::HasSamePath(track.fPos, *track.fGeometryState.fPath, *tmpstate);
  if (!samepath) {
    tmpstate->CopyTo(track.fGeometryState.fNextpath);
#ifdef VECGEOM_CACHED_TRANS
    track.fGeometryState.fNextpath->UpdateTopMatrix();
#endif
  }
  same = samepath;
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geant
