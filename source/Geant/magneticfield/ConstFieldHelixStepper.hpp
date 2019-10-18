/*
 * ConstFieldHelixStepper.hpp
 *
 *
 *  Started from Const[Bz]FieldHelixStepper by S. Wenzel and J. Apostolakis
 */

#pragma once

#include <base/Vector3D.h>
#include <Geant/core/Config.hpp>
#include <Geant/core/VectorTypes.hpp>

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
/**
 * A very simple stepper treating the propagation of particles in a constant Bz magnetic field
 * ( neglecting energy loss of particle )
 * This class is roughly equivalent to TGeoHelix in ROOT
 */
class ConstFieldHelixStepper {
  template <typename T>
  using Vector3D = vecgeom::Vector3D<T>;

public:
  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(double Bx, double By, double Bz);

  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(double Bfield[3]);

  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(Vector3D<double> const &Bfield);

  void SetB(double Bx, double By, double Bz)
  {
    fB.Set(Bx, By, Bz);
    CalculateDerived();
  }
  Vector3D<double> const &GetB() const { return fB; }

  /*
  template<typename RT, typename Vector3D>
  RT GetCurvature(Vector3D const & dir,
                  double const charge, double const momentum) const
  {
    if (charge == 0) return RT(0.);
    return abs( kB2C * fBz * dir.FastInverseScaledXYLength( momentum ) );
  }
  */

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Real_v>
  GEANT_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoStep(Real_v const &posx, Real_v const &posy, Real_v const &posz,
                                                         Real_v const &dirx, Real_v const &diry, Real_v const &dirz,
                                                         Real_v const &charge, Real_v const &momentum,
                                                         Real_v const &step, Real_v &newposx, Real_v &newposy,
                                                         Real_v &newposz, Real_v &newdirx, Real_v &newdiry,
                                                         Real_v &newdirz) const;

  /**
   * basket version of dostep
   * version that takes plain arrays as input; suited for first-gen GeantV
   */
  template <typename Real_v>
  GEANT_FORCE_INLINE void DoStepArr(double const *posx, double const *posy, double const *posz, double const *dirx,
                                    double const *diry, double const *dirz, double const *charge,
                                    double const *momentum, double const *step, double *newposx, double *newposy,
                                    double *newposz, double *newdirx, double *newdiry, double *newdirz, int np) const;


  // in future will offer versions that take containers as input

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Real_v>
  GEANT_FORCE_INLINE void DoStep(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                 Real_v const &charge, Real_v const &momentum, Real_v const &step,
                                 Vector3D<Real_v> &newPosition, Vector3D<Real_v> &newDirection) const;

  // Auxiliary methods
  template <typename Real_v>
  GEANT_FORCE_INLINE void PrintStep(vecgeom::Vector3D<Real_v> const &startPosition,
                                    vecgeom::Vector3D<Real_v> const &startDirection, Real_v const &charge,
                                    Real_v const &momentum, Real_v const &step, vecgeom::Vector3D<Real_v> &endPosition,
                                    vecgeom::Vector3D<Real_v> &endDirection) const;

protected:
  void CalculateDerived();

  template <typename Real_v>
  GEANT_FORCE_INLINE bool CheckModulus(Real_v &newdirX_v, Real_v &newdirY_v, Real_v &newdirZ_v) const;

private:
  Vector3D<double> fB;
  // Auxilary members - calculated from above - cached for speed, code simplicity
  double fBmag;
  Vector3D<double> fUnit;
}; // end class declaration

GEANT_FORCE_INLINE
void ConstFieldHelixStepper::CalculateDerived()
{
  fBmag = fB.Mag();
  fUnit = fB.Unit();
}

GEANT_FORCE_INLINE
ConstFieldHelixStepper::ConstFieldHelixStepper(double Bx, double By, double Bz) : fB(Bx, By, Bz)
{
  CalculateDerived();
}

GEANT_FORCE_INLINE
ConstFieldHelixStepper::ConstFieldHelixStepper(double B[3]) : fB(B[0], B[1], B[2])
{
  CalculateDerived();
}

GEANT_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
ConstFieldHelixStepper::ConstFieldHelixStepper(vecgeom::Vector3D<double> const &Bfield) : fB(Bfield)
{
  CalculateDerived();
}

/**
 * this function propagates the track along the "helix-solution" by a step step
 * input: current position (x0, y0, z0), current direction ( dirX0, dirY0, dirZ0 ), some particle properties
 * output: new position, new direction of particle
 */
template <typename Real_v>
GEANT_FORCE_INLINE void ConstFieldHelixStepper::DoStep(Real_v const &x0, Real_v const &y0, Real_v const &z0,
                                                       Real_v const &dirX0, Real_v const &dirY0, Real_v const &dirZ0,
                                                       Real_v const &charge, Real_v const &momentum, Real_v const &step,
                                                       Real_v &x, Real_v &y, Real_v &z, Real_v &dx, Real_v &dy,
                                                       Real_v &dz) const
{
  vecgeom::Vector3D<Real_v> startPosition(x0, y0, z0);
  vecgeom::Vector3D<Real_v> startDirection(dirX0, dirY0, dirZ0);
  vecgeom::Vector3D<Real_v> endPosition, endDirection;

  // startPosition.Set( x0, y0, z0);
  // startDirection.Set( dirX0, dirY0, dirZ0);

  DoStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
  x  = endPosition.x();
  y  = endPosition.y();
  z  = endPosition.z();
  dx = endDirection.x();
  dy = endDirection.y();
  dz = endDirection.z();

  // PrintStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
}

template <typename Real_v>
GEANT_FORCE_INLINE void ConstFieldHelixStepper::DoStep(vecgeom::Vector3D<Real_v> const &startPosition,
                                                       vecgeom::Vector3D<Real_v> const &startDirection,
                                                       Real_v const &charge, Real_v const &momentum, Real_v const &step,
                                                       vecgeom::Vector3D<Real_v> &endPosition,
                                                       vecgeom::Vector3D<Real_v> &endDirection) const
{
  const Real_v kB2C_local(-0.299792458e-3);
  const Real_v kSmall(1.E-30);
  using vecCore::math::Max;
  using vecCore::math::SinCos;
  using vecgeom::Vector3D;
  // using vecCore::math::Sin;
  // using vecCore::math::Cos;
  using vecCore::math::Abs;
  using vecCore::math::Sqrt;
  // could do a fast square root here

  // Real_v dt = Sqrt((dx0*dx0) + (dy0*dy0)) + kSmall;

  // std::cout << " ConstFieldHelixStepper::DoStep called.  fBmag= " << fBmag
  //          << " unit dir= " << fUnit << std::endl;

  // assert( std::abs( startDirection.Mag2() - 1.0 ) < 1.0e-6 );

  Vector3D<Real_v> dir1Field(fUnit);
  Real_v UVdotUB = startDirection.Dot(dir1Field); //  Limit cases 0.0 and 1.0
  Real_v dt2     = Max(startDirection.Mag2() - UVdotUB * UVdotUB, Real_v(0.0));
  Real_v sinVB   = Sqrt(dt2) + kSmall;

  // Real_v invnorm = 1. / sinVB;

  // radius has sign and determines the sense of rotation
  Real_v R = momentum * sinVB / (kB2C_local * charge * fBmag);

  Vector3D<Real_v> restVelX = startDirection - UVdotUB * dir1Field;

  // Vector3D<Real_v>  dirVelX( 0.0, 0.0, 0.0 );            // OK if it is zero - ie. dir // B
  // if( restVelX.Mag2() > 0.0 ) dirVelX = restVelX.Unit();
  Vector3D<Real_v> dirVelX    = restVelX.Unit();          // Unit must cope with 0 length !!
  Vector3D<Real_v> dirCrossVB = dirVelX.Cross(dir1Field); // OK if it is zero
  // Vector3D<Real_v>  dirCrossVB= restVelX.Cross(dir1Field);  // OK if it is zero

  /***
  printf("\n");
  printf("CVFHS> dir-1  B-fld  = %f %f %f   mag-1= %g \n", dir1Field.x(), dir1Field.y(), dir1Field.z(),
         dir1Field.Mag()-1.0 );
  printf("CVFHS> dir-2  VelX   = %f %f %f   mag-1= %g \n", dirVelX.x(), dirVelX.y(), dirVelX.z(),
         dirVelX.Mag()-1.0 );
  printf("CVFHS> dir-3: CrossVB= %f %f %f   mag-1= %g \n", dirCrossVB.x(), dirCrossVB.y(), dirCrossVB.z(),
         dirCrossVB.Mag()-1.0 );
  // dirCrossVB = dirCrossVB.Unit();
  printf("CVFHS> Dot products   d1.d2= %g   d2.d3= %g  d3.d1= %g \n",
         dir1Field.Dot(dirVelX), dirVelX.Dot( dirCrossVB), dirCrossVB.Dot(dir1Field) );
   ***/
  assert(vecCore::MaskFull(Abs(dir1Field.Dot(dirVelX)) < 1.e-6));
  assert(vecCore::MaskFull(Abs(dirVelX.Dot(dirCrossVB)) < 1.e-6));
  assert(vecCore::MaskFull(Abs(dirCrossVB.Dot(dir1Field)) < 1.e-6));

  Real_v phi = -step * charge * fBmag * kB2C_local / momentum;

  // printf("CVFHS> phi= %g \n", vecCore::Get(phi,0) );  // phi (scalar)  or phi[0] (vector)

  Real_v cosphi;                 //  = Cos(phi);
  Real_v sinphi;                 //  = Sin(phi);
  SinCos(phi, &sinphi, &cosphi); // Opportunity for new 'efficient' method !?

  endPosition = startPosition + R * (cosphi - 1) * dirCrossVB - R * sinphi * dirVelX +
                step * UVdotUB * dir1Field; //   'Drift' along field direction

  // dx = dx0 * cosphi - sinphi * dy0;
  // dy = dy0 * cosphi + sinphi * dx0;
  // dz = dz0;
  // printf(" phi= %f, sin(phi)= %f , sin(V,B)= %f\n", phi, sinphi, sinVB );
  endDirection = UVdotUB * dir1Field + cosphi * sinVB * dirVelX + sinphi * sinVB * dirCrossVB;
}

/**
 * basket version of dostep
 */

//  SW: commented out due to explicit Vc dependence and since it is not currently used
//       leaving the code here to show how one would dispatch to the kernel with Vc
#define _R_ __restrict__

template <typename Real_v>
void ConstFieldHelixStepper::DoStepArr(double const *_R_ posx, double const *_R_ posy, double const *_R_ posz,
                                       double const *_R_ dirx, double const *_R_ diry, double const *_R_ dirz,
                                       double const *_R_ charge, double const *_R_ momentum, double const *_R_ step,
                                       double *_R_ newposx, double *_R_ newposy, double *_R_ newposz,
                                       double *_R_ newdirx, double *_R_ newdiry, double *_R_ newdirz, int np) const
{
  const size_t vectorSize = vecCore::VectorSize<Real_v>();
  using vecCore::Load;
  using vecCore::Set;
  using vecCore::Store;

  // std::cout << " --- ConstFieldHelixStepper::DoStepArr called." << std::endl;

  int i;
  for (i = 0; i < np; i += vectorSize) {
    // results cannot not be temporaries
    //    Vector3D<Real_v> newPosition, newDirection;
    Real_v oldPosx_v, oldPosy_v, oldPosz_v, oldDirx_v, oldDiry_v, oldDirz_v;
    Real_v newposx_v, newposy_v, newposz_v, newdirx_v, newdiry_v, newdirz_v;
    Real_v charge_v, momentum_v, stepSz_v;

    Load(oldPosx_v, &posx[i]);
    Load(oldPosy_v, &posy[i]);
    Load(oldPosz_v, &posz[i]);
    Load(oldDirx_v, &dirx[i]);
    Load(oldDiry_v, &diry[i]);
    Load(oldDirz_v, &dirz[i]);
    Load(charge_v, &charge[i]);
    Load(momentum_v, &momentum[i]);
    Load(stepSz_v, &step[i]);

    // This check should be optional
    CheckModulus<Real_v>(oldDirx_v, oldDiry_v, oldDirz_v);

    DoStep<Real_v>(oldPosx_v, oldPosy_v, oldPosz_v, oldDirx_v, oldDiry_v, oldDirz_v, charge_v, momentum_v, stepSz_v,
                   newposx_v, newposy_v, newposz_v, newdirx_v, newdiry_v, newdirz_v);

    CheckModulus(newdirx_v, newdiry_v, newdirz_v);

    // write results
    Store(newposx_v, &newposx[i]);
    Store(newposy_v, &newposy[i]);
    Store(newposz_v, &newposz[i]);
    Store(newdirx_v, &newdirx[i]);
    Store(newdiry_v, &newdiry[i]);
    Store(newdirz_v, &newdirz[i]);
  }

  // tail part
  for (; i < np; i++)
    DoStep<double>(posx[i], posy[i], posz[i], dirx[i], diry[i], dirz[i], charge[i], momentum[i], step[i], newposx[i],
                   newposy[i], newposz[i], newdirx[i], newdiry[i], newdirz[i]);
}

//________________________________________________________________________________
template <typename Real_v>
GEANT_FORCE_INLINE bool ConstFieldHelixStepper::CheckModulus(Real_v &newdirX_v, Real_v &newdirY_v,
                                                             Real_v &newdirZ_v) const
{
  constexpr double perMillion = 1.0e-6;

  Real_v modulusDir = newdirX_v * newdirX_v + newdirY_v * newdirY_v + newdirZ_v * newdirZ_v;
  typename vecCore::Mask<Real_v> goodDir;
  goodDir = vecCore::math::Abs(modulusDir - Real_v(1.0)) < perMillion;

  bool allGood = vecCore::MaskFull(goodDir);
  assert(allGood && "Not all Directions are nearly 1");

  return allGood;
}

template <typename Real_v>
GEANT_FORCE_INLINE void ConstFieldHelixStepper::PrintStep(vecgeom::Vector3D<Real_v> const &startPosition,
                                                          vecgeom::Vector3D<Real_v> const &startDirection,
                                                          Real_v const &charge, Real_v const &momentum,
                                                          Real_v const &step, vecgeom::Vector3D<Real_v> &endPosition,
                                                          vecgeom::Vector3D<Real_v> &endDirection) const
{
  // Debug printing of input & output
  using vecCore::Get;
  printf(" HelixSteper::PrintStep \n");
  const int vectorSize = vecCore::VectorSize<Real_v>();
  Real_v x0, y0, z0, dirX0, dirY0, dirZ0;
  Real_v x, y, z, dx, dy, dz;
  x0    = startPosition.x();
  y0    = startPosition.y();
  z0    = startPosition.z();
  dirX0 = startDirection.x();
  dirY0 = startDirection.y();
  dirZ0 = startDirection.z();
  x     = endPosition.x();
  y     = endPosition.y();
  z     = endPosition.z();
  dx    = endDirection.x();
  dy    = endDirection.y();
  dz    = endDirection.z();
  for (int i = 0; i < vectorSize; i++) {
    printf("Start> Lane= %1d Pos= %8.5f %8.5f %8.5f  Dir= %8.5f %8.5f %8.5f ", i, Get(x0, i), Get(y0, i), Get(z0, i),
           Get(dirX0, i), Get(dirY0, i), Get(dirZ0, i));
    printf(" s= %10.6f ", Get(step, i) / 10.0);     // / units::mm );
    printf(" q= %3.1f ", Get(charge, i) / 10.0);    // in e+ units ?
    printf(" p= %10.6f ", Get(momentum, i) / 10.0); // / units::GeV );
    // printf(" ang= %7.5f ", angle );
    printf(" End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", Get(x, i), Get(y, i), Get(z, i), Get(dx, i),
           Get(dy, i), Get(dz, i));
  }
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geant

