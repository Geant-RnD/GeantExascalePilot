#include "Geant/magneticfield/FieldPropagationHandler.hpp"

#include "Geant/track/TrackState.hpp"

#include <sstream>
#include <ostream>
#include <iomanip>

#include "Geant/core/VectorTypes.hpp" // Defines geant::Double_v etc
#include "Geant/core/SystemOfUnits.hpp"

//#include "Geant/WorkspaceForFieldPropagation.h"

#include "navigation/NavigationState.h"

#include "Geant/magneticfield/ConstFieldHelixStepper.hpp"
#include "Geant/geometry/NavigationInterface.hpp"

#if 0
#  include "Geant/ScalarNavInterfaceVG.h"
#  include "Geant/ScalarNavInterfaceVGM.h"
#  include "Geant/VectorNavInterface.h"
#endif

using Double_v = geantx::Double_v;

// #define CHECK_VS_RK   1
// #define CHECK_VS_HELIX 1

// #define REPORT_AND_CHECK 1

// #define STATS_METHODS 1
// #define DEBUG_FIELD 1

#ifdef CHECK_VS_HELIX
#  define CHECK_VS_SCALAR 1
#endif

#ifdef CHECK_VS_RK
#  define CHECK_VS_SCALAR 1
#endif

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

constexpr double gEpsDeflection = 1.E-2 * units::cm;

constexpr auto stageAfterCrossing = SimulationStage::PostPropagationStage;

static constexpr double kB2C = -0.299792458e-3;

//______________________________________________________________________________
// Curvature for general field
VECCORE_ATT_HOST_DEVICE
double FieldPropagationHandler::Curvature(const TrackState &track) const
{

  ThreeVector_t magFld;
  double bmag = 0.0;

  ThreeVector_t Position(track.fPos);
  FieldLookup::GetFieldValue(Position, magFld, bmag); // , td);

  return Curvature(track, magFld, bmag);
}

// Needing a real implementation.

double Charge(const TrackState &track)
{
  return 1;
}

//______________________________________________________________________________
// Curvature for general field
VECCORE_ATT_HOST_DEVICE
double FieldPropagationHandler::Curvature(const TrackState &track,
                                          const ThreeVector_t &magFld_in,
                                          double bmag) const
{
  // using ThreeVector_t            = vecgeom::Vector3D<double>;
  constexpr double tiny          = 1.E-30;
  constexpr double inv_kilogauss = 1.0 / units::kilogauss;

  bmag *= inv_kilogauss;
  ThreeVector_t magFld{magFld_in * inv_kilogauss};
  //  Calculate transverse momentum 'Pt' for field 'B'
  //
  ThreeVector_t Momentum(track.fDir);
  ThreeVector_t PtransB; //  Transverse wrt direction of B
  double ratioOverFld = 0.0;
  if (bmag > 0) ratioOverFld = Momentum.Dot(magFld) / (bmag * bmag);
  PtransB       = Momentum - ratioOverFld * magFld;
  double Pt_mag = PtransB.Mag();

  return fabs(kB2C * Charge(track) * bmag / (Pt_mag + tiny));
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void FieldPropagationHandler::DoIt(TrackState &track, TaskData *td) const
{
  // Scalar geometry length computation. The track is moved into the output basket.
  using vecCore::math::Max;
  using vecCore::math::Min;
  constexpr double step_push = 1.e-4;
  // The minimum step is step_push in case the physics step limit is not smaller
  double step_min = Min(track.fPhysicsState.fPstep, step_push);
  // The track snext value is already the minimum between geometry and physics
  double step_geom_phys = Max(step_min, track.fGeometryState.fSnext);
  // Field step limit. We use the track sagitta to estimate the "bending" error,
  // i.e. what is the propagated length for which the track deviation in
  // magnetic field with respect to straight propagation is less than epsilon.
  double bmag = -1.0;
  ThreeVector BfieldInitial;
  ThreeVector Position(track.fPos);
  FieldLookup::GetFieldValue(Position, BfieldInitial, bmag);
  double step_field = Max(SafeLength(track, gEpsDeflection, BfieldInitial, bmag),
                          track.fGeometryState.fSafety);

  double step = Min(step_geom_phys, step_field);

  // Propagate in magnetic field
  PropagateInVolume(track, step, BfieldInitial, bmag, td);
  // Update number of partial steps propagated in field
  // td->fNmag++;

#ifndef IMPLEMENTED_STATUS
#  warning "Propagation has no way to tell scheduler what to do yet."
#else
  // Set continuous processes stage as follow-up for tracks that reached the
  // physics process
  if (track.fStatus() == SimulationStage::Physics) {
    // Update number of steps to physics and total number of steps
    // td->fNphys++;
    // td->fNsteps++;
    track->SetStage(stageAfterCrossing); // Future: (kPostPropagationStage);
  } else {
    // Crossing tracks continue to continuous processes, the rest have to
    // query again the geometry
    if ((track->GetSafety() < 1.E-10) && !IsSameLocation(*track, td)) {
      // td->fNcross++;
      // td->fNsteps++;
    } else {
      track->SetStage(SimulationStage::GeometryStepStage);
    }
  }
#endif
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void FieldPropagationHandler::PropagateInVolume(TrackState &track, double crtstep,
                                                const ThreeVector &BfieldInitial,
                                                double bmag, TaskData *td) const
{
  // Single track propagation in a volume. The method is to be called
  // only with  charged tracks in magnetic field.The method decreases the fPstepV
  // fSafetyV and fSnextV with the propagated values while increasing the fStepV.
  // The status and boundary flags are set according to which gets hit first:
  // - physics step (bdr=0)
  // - safety step (bdr=0)
  // - snext step (bdr=1)

  // std::cout << "FieldPropagationHandler::PropagateInVolume called for 1 track" <<
  // std::endl;

  constexpr double toKiloGauss = 1.0 / units::kilogauss; // Converts to kilogauss

#if ENABLE_MORE_COMPLEX_FIELD
  bool useRungeKutta   = td->fPropagator->fConfig->fUseRungeKutta;
  auto fieldConfig     = FieldLookup::GetFieldConfig();
  auto fieldPropagator = td->fFieldPropagator;
#endif

  /****
  #ifndef VECCORE_CUDA_DEVICE_COMPILATION
    auto fieldPropagator = GetFieldPropagator(td);
    if (!fieldPropagator && !td->fSpace4FieldProp) {
      fieldPropagator = Initialize(td);
    }
  #endif
  *****/

#if DEBUG_FIELD
  bool verboseDiff = true; // If false, print just one line.  Else more details.
  bool epsilonRK   = td->fPropagator->fConfig->fEpsilonRK;
#endif

#ifdef PRINT_STEP_SINGLE
  double curvaturePlus = fabs(kB2C * track.Charge() * (bmag * toKiloGauss)) /
                         (track.P() + 1.0e-30); // norm for step
  const double angle = crtstep * curvaturePlus;
  Print("--PropagateInVolume(Single): ",
        "Momentum= %9.4g (MeV) Curvature= %9.4g (1/mm)  CurvPlus= %9.4g (1/mm)  step= "
        "%f (mm)  Bmag=%8.4g KG   angle= %g\n",
        (track.P() / units::MeV), Curvature(track) * units::mm, curvaturePlus * units::mm,
        crtstep / units::mm, bmag * toKiloGauss, angle);
// Print("\n");
#endif

  const ThreeVector_t Position(track.fPos);
  const ThreeVector_t Direction(track.fDir);
  ThreeVector_t PositionNew(0., 0., 0.);
  ThreeVector_t DirectionNew(0., 0., 0.);

  // char method= '0';
  ThreeVector_t PositionNewCheck(0., 0., 0.);
  ThreeVector_t DirectionNewCheck(0., 0., 0.);

#if ENABLE_MORE_COMPLEX_FIELD
  if (useRungeKutta || !fieldConfig->IsFieldUniform()) {
    assert(fieldPropagator);
    fieldPropagator->DoStep(Position, Direction, track.Charge(), track.P(), crtstep,
                            PositionNew, DirectionNew);
    assert((PositionNew - Position).Mag() < crtstep + 1.e-4);
#  ifdef DEBUG_FIELD
// cross check
#    ifndef CHECK_VS_BZ
    ConstFieldHelixStepper stepper(BfieldInitial * toKiloGauss);
    stepper.DoStep<double>(Position, Direction, track.Charge(), track.P(), crtstep,
                           PositionNewCheck, DirectionNewCheck);
#    else
    double Bz = BfieldInitial[2] * toKiloGauss;
    ConstBzFieldHelixStepper stepper_bz(Bz); //
    stepper_bz.DoStep<ThreeVector, double, int>(Position, Direction, track.Charge(),
                                                track.P(), crtstep, PositionNewCheck,
                                                DirectionNewCheck);
#    endif

    double posShift = (PositionNew - PositionNewCheck).Mag();
    double dirShift = (DirectionNew - DirectionNewCheck).Mag();

    if (posShift > epsilonRK || dirShift > epsilonRK) {
      std::cout << "*** position/direction shift RK vs. HelixConstBz :" << posShift
                << " / " << dirShift << "\n";
      if (verboseDiff) {
        printf("%s End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n",
               " FPH::PiV(1)-RK: ", PositionNew.x(), PositionNew.y(), PositionNew.z(),
               DirectionNew.x(), DirectionNew.y(), DirectionNew.z());
        printf("%s End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n",
               " FPH::PiV(1)-Bz: ", PositionNewCheck.x(), PositionNewCheck.y(),
               PositionNewCheck.z(), DirectionNewCheck.x(), DirectionNewCheck.y(),
               DirectionNewCheck.z());
      }
    }
#  endif
// method= 'R';
#  ifdef STATS_METHODS
    numRK++;
    numTot++;
#  endif
  } else {
#endif
    // geant::
    double BfieldArr[3] = {BfieldInitial.x() * toKiloGauss,
                           BfieldInitial.y() * toKiloGauss,
                           BfieldInitial.z() * toKiloGauss};
    ConstFieldHelixStepper stepper(BfieldArr);
    stepper.DoStep<double>(Position, Direction, Charge(track),
                           track.fPhysicsState.fMomentum, crtstep, PositionNew,
                           DirectionNew);
// method= 'v';
#ifdef STATS_METHODS
    numHelixGen++;
    numTot++;
#endif
#if ENABLE_MORE_COMPLEX_FIELD
  }
#endif

#ifdef PRINT_FIELD
  // Print(" FPH::PiV(1): Start>", " Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f",
  // Position.x(), Position.y(), Position.z(), Direction.x(), Direction.y(), Direction.z()
  // ); Print(" FPH::PiV(1): End>  ", " Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f",
  // PositionNew.x(), PositionNew.y(), PositionNew.z(), DirectionNew.x(),
  // DirectionNew.y(), DirectionNew.z() );

  // printf(" FPH::PiV(1): ");
  printf(" FPH::PiV(1):: ev= %3d trk= %3d %3d %c ", track.Event(), track.Particle(),
         track.GetNsteps(), method);
  printf("Start> Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f ", Position.x(),
         Position.y(), Position.z(), Direction.x(), Direction.y(), Direction.z());
  printf(" s= %10.6f ang= %7.5f ", crtstep / units::mm, angle);
  printf( // " FPH::PiV(1): "
      "End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", PositionNew.x(),
      PositionNew.y(), PositionNew.z(), DirectionNew.x(), DirectionNew.y(),
      DirectionNew.z());
#endif

  // std::cout << " total calls: " << numTot << std::endl;
#ifdef STATS_METHODS
  unsigned long ntot = numTot;
  if (ntot % gPrintStatsMod < 1) {
    PrintStats();
    // if (numTot > 10 * gPrintStatsMod) gPrintStatsMod = 10 * gPrintStatsMod;
  }
#endif

  //  may normalize direction here  // vecCore::math::Normalize(dirnew);
  ThreeVector DirectionUnit = DirectionNew.Unit();
  double posShiftSq         = (PositionNew - Position).Mag2();

#ifdef COMPLETE_FUNCTIONAL_UPDATES
  track.SetPosition(PositionNew);
  track.SetDirection(DirectionUnit);
  track.NormalizeFast();

  // Reset relevant variables
  track.SetStatus(kInFlight);
  track.IncrementNintSteps();
  track.IncreaseStep(crtstep);

  track.DecreasePstep(crtstep);
  if (track.fPhysicsState.fPstep < 1.E-10) {
    track.SetPstep(0);
    track.SetStatus(kPhysics);
  }
  track.DecreaseSnext(crtstep);
  if (track.GetSnext() < 1.E-10) {
    track.SetSnext(0);
    if (track.Boundary()) track.SetStatus(kBoundary);
  }

  double preSafety = track.GetSafety();
  if (posShiftSq > preSafety * preSafety) {
    track.SetSafety(0);
  } else {
    double posShift = std::sqrt(posShiftSq);
    track.DecreaseSafety(posShift);
    if (track.GetSafety() < 1.E-10) track.SetSafety(0);
  }
#else
  track.fPos = PositionNew;
  track.fDir = DirectionUnit;
  // TODO: need to add normlization of track.
  // Normalize(track)

  // track.SetStatus(kInFlight);
  track.fStep += crtstep;

  track.fPhysicsState.fPstep -= crtstep;
  if (track.fPhysicsState.fPstep < 1.E-10) {
    track.fPhysicsState.fPstep = 0;
    // track.SetStatus(kPhysics);
  }

  track.fGeometryState.fSnext -= crtstep;
  if (track.fGeometryState.fSnext < 1.E-10) {
    track.fGeometryState.fSnext = 0;
    // if (track.Boundary()) track.SetStatus(kBoundary);
  }

  const auto preSafety = track.fGeometryState.fSafety;
  if (posShiftSq > preSafety * preSafety) {
    track.fGeometryState.fSafety = 0;
  } else {
    double posShift = std::sqrt(posShiftSq);
    track.fGeometryState.fSafety -= posShift;
    if (track.fGeometryState.fSafety < 1.E-10) track.fGeometryState.fSafety = 0;
  }

#endif

#ifdef REPORT_AND_CHECK
  CheckTrack(track, "End of Propagate-In-Volume", 1.0e-5);
#endif
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool FieldPropagationHandler::IsSameLocation(TrackState &track, TaskData *td) const
{
  // Query geometry if the location has changed for a track
  // Returns number of tracks crossing the boundary (0 or 1)

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

  delete tmpstate;

  if (same) {
    track.fGeometryState.fBoundary = false;
    return true;
  }

  track.fGeometryState.fBoundary = true;
  // track.SetStatus(kBoundary);
  // if (track.NextPath()->IsOutside())
  //    track.SetStatus(kExitingSetup);

  // if (track.GetStep() < 1.E-8) td->fNsmall++;
  return false;
}

#ifdef REPORT_AND_CHECK
#  define IsNan(x) (!(x > 0 || x <= 0.0))
//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void FieldPropagationHandler::CheckTrack(TrackState &track, const char *msg,
                                         double epsilon) const
{
  // Ensure that values are 'sensible' - else print msg and track
  if (epsilon <= 0.0 || epsilon > 0.01) {
    epsilon = 1.e-6;
  }

  double x = track.X(), y = track.Y(), z = track.Z();
  bool badPosition       = IsNan(x) || IsNan(y) || IsNan(z);
  const double maxRadius = 10000.0; // Should be a property of the geometry
  const double maxRadXY  = 5000.0;  // Should be a property of the geometry

  // const double maxUnitDev =  1.0e-4;  // Deviation from unit of the norm of the
  // direction
  double radiusXy2 = x * x + y * y;
  double radius2   = radiusXy2 + z * z;
  badPosition      = badPosition || (radiusXy2 > maxRadXY * maxRadXY) ||
                (radius2 > maxRadius * maxRadius);

  const double maxUnitDev =
      epsilon; // Use epsilon for max deviation of direction norm from 1.0

  double dx = track.Dx(), dy = track.Dy(), dz = track.Dz();
  double dirNorm2   = dx * dx + dy * dy + dz * dz;
  bool badDirection = std::fabs(dirNorm2 - 1.0) > maxUnitDev;
  if (badPosition || badDirection) {
    static const char *errMsg[4] = {" All ok - No error. ",
                                    " Bad position.",                 // [1]
                                    " Bad direction.",                // [2]
                                    " Bad direction and position. "}; // [3]
    int iM                       = 0;
    if (badPosition) {
      iM++;
    }
    if (badDirection) {
      iM += 2;
    }
    // if( badDirection ) {
    //   Printf( " Norm^2 direction= %f ,  Norm -1 = %g", dirNorm2, sqrt(dirNorm2)-1.0 );
    // }
    Printf("ERROR> Problem with track %p . Issue: %s. Info message: %s -- Mag^2(dir)= "
           "%9.6f Norm-1= %g",
           (void *)&track, errMsg[iM], msg, dirNorm2, sqrt(dirNorm2) - 1.0);
    track.Print(msg);
  }
}
#endif

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
