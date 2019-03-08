/*

Cell-version. Takes 4 times more memory for storing input magnetic field map.
Every point stored 4 times for the 4 blocks it is counted towards.
MagCellStructs stores value of mag. field for i1,i2,i3,i4


*/

#ifndef _MAGFIELDCELLVERSION_H_
#define _MAGFIELDCELLVERSION_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/Global.h"
#include <Vc/Vc>
#include "backend/vc/Backend.h"
#include "backend/vcfloat/Backend.h"
#include "backend/scalarfloat/Backend.h"
#include "backend/Backend.h"
#include <cassert>
#include <ctime>
using namespace std;

// typedef double dataType;
typedef float dataType;

#define FAST

#ifdef FAST
#define INLINE_CHOICE inline __attribute__((always_inline))
#endif
#ifndef FAST
#define INLINE_CHOICE __attribute__((noinline))
#endif

#ifndef _MAGFIELD_H_
#include "Geant/Structs.h"
#endif

class MagFieldCellVersion {
public:
  MagFieldCellVersion();

  // New stuff
  // Takes as input x,y,z; Gives output Bx,By,Bz
  template <class Backend>
  void GetFieldValue(const vecgeom::Vector3D<typename Backend::precision_v> &pos,
                     vecgeom::Vector3D<typename Backend::precision_v> &xyzField);

  // Reads data from given 2D magnetic field map. Can be easily modified to read a given 2D map, in case the file
  // changes
  void ReadVectorData(std::string inputMap);

  void StoreDataInMag4();

  ~MagFieldCellVersion();

private:
  //  Invariants -- parameters of the field
  const dataType millimeter = 1.0; // Currently -- to be native GeantV

  const dataType kRMax   = 9000. * millimeter;  //  Maximum value of R =  9.00 meters
  const dataType kZMax   = 16000. * millimeter; //  Max value of Z = 16.00 meters
  const int kNoZValues   = 161;
  const int kNoRValues   = 181;
  const int kHalfZValues = 80;

  // Derived values
  // kRDiff and kZDiff take care of mm because they come from kRMax and kZMax which have mm in them
  const dataType kRDiff = kRMax / (kNoRValues - 1);     //  Radius increment between lattice points
  const dataType kZDiff = 2 * kZMax / (kNoZValues - 1); //  Z increment

  const dataType kZ0       = -kZMax;
  const dataType kRDiffInv = 1.f / kRDiff;
  const dataType kZDiffInv = 1.f / kZDiff;
  const dataType kAInverse = 1 / (kRDiff * kZDiff);

  // For (R,Z) pairs : gives field in cylindrical coordinates in rzfield
  template <class Backend>
  void GetFieldValueRZ(const typename Backend::precision_v &radius, const typename Backend::precision_v &z,
                       vecgeom::Vector3D<typename Backend::precision_v> &rzField);

  // Used to convert cartesian coordinates to cylindrical coordinates R-Z-phi
  // Does not calculate phi
  template <class Backend>
  void CartesianToCylindrical(const vecgeom::Vector3D<typename Backend::precision_v> &cart,
                              typename Backend::precision_v cyl[2]);

  // Converts cylindrical magnetic field to field in cartesian coordinates
  template <class Backend>
  void CylindricalToCartesian(const vecgeom::Vector3D<typename Backend::precision_v> &rzField,
                              const typename Backend::precision_v sinTheta,
                              const typename Backend::precision_v cosTheta,
                              vecgeom::Vector3D<typename Backend::precision_v> &xyzField);

  // Takes care of indexing into multiple places in AOS. Gather because using gather
  // defined in Vc class. Not self-defined gather like before
  template <class Backend>
  void Gather(const typename Backend::precision_v index, typename Backend::precision_v B[3]);

  template <class Backend>
  void NewGather(const typename Backend::precision_v index, typename Backend::precision_v B1[3],
                 typename Backend::precision_v B2[3], typename Backend::precision_v B3[3],
                 typename Backend::precision_v B4[3]);

private:
  MagVector *fMagVector           = new MagVector[30000];
  MagCellStructs *fMagCellStructs = new MagCellStructs[30000];
};

MagFieldCellVersion::MagFieldCellVersion()
{
  std::cout << "Cell-version" << std::endl;
}

MagFieldCellVersion::~MagFieldCellVersion()
{
}

INLINE_CHOICE
void MagFieldCellVersion::ReadVectorData(std::string inputMap)
{
  std::string line;
  std::string s1, s2, s3, s4, s5, s0;
  dataType d1, d2, d3, d4, d5, d0;
  int ind = 0;
  ifstream pFile(inputMap);
  if (pFile.is_open()) {
    // getline() returns the stream. testing the stream with while returns error such as EOF
    while (getline(pFile, line)) {
      // so here we know that the read was a success and that line has valid data
      stringstream ss(line);
      // parsing all the parts. s0's store the string names which are of no use to us.
      ss >> s0 >> d1 >> s1 >> d0 >> s2 >> d2 >> s3 >> d3 >> s4 >> d4 >> s5 >> d5;

      fMagVector[ind].SetBr(d4 * kAInverse);
      fMagVector[ind].SetBphi(d5 * kAInverse);
      fMagVector[ind].SetBz(d3 * kAInverse);
      ind++;
    }
    pFile.close();

    StoreDataInMag4();
  } else {
    cout << "Unable to open file";
  }
}

INLINE_CHOICE
void MagFieldCellVersion::StoreDataInMag4()
{
  // I have fMagCellStructs[30000]
  // j from 0 to 3
  // i from 0 to 29141 - 161/181
  for (int i = 0; i < (kNoRValues - 1) * kNoZValues; ++i) {
    fMagCellStructs[i].m0 = fMagVector[i];
    fMagCellStructs[i].m1 = fMagVector[i + 1];
    fMagCellStructs[i].m2 = fMagVector[i + kNoZValues];
    fMagCellStructs[i].m3 = fMagVector[i + kNoZValues + 1];
  }
}

template <class Backend>
INLINE_CHOICE void MagFieldCellVersion::CartesianToCylindrical(
    const vecgeom::Vector3D<typename Backend::precision_v> &cart, typename Backend::precision_v cyl[2])
{

  // cyl[] =[r,z]
  cyl[0] = sqrt(cart.x() * cart.x() + cart.y() * cart.y()); // r = sqrt(x^2 + y^2)
  cyl[1] = cart.z();                                        // z = z
}

template <class Backend>
INLINE_CHOICE void MagFieldCellVersion::CylindricalToCartesian(
    const vecgeom::Vector3D<typename Backend::precision_v> &rzField, const typename Backend::precision_v sinTheta,
    const typename Backend::precision_v cosTheta, vecgeom::Vector3D<typename Backend::precision_v> &xyzField)
{
  // rzField[] has r, phi and z

  xyzField.x() = rzField.x() * cosTheta - rzField.y() * sinTheta; // Bx= Br cos(theta) - Bphi sin(theta)
  xyzField.y() = rzField.x() * sinTheta + rzField.y() * cosTheta; // By = Br sin(theta) + Bphi cos(theta)
  xyzField.z() = rzField.z();                                     // Bz = Bz
}

// Scalar Backend method
template <class Backend>
INLINE_CHOICE void MagFieldCellVersion::Gather(const typename Backend::precision_v index,
                                               typename Backend::precision_v B[3])
{

  int intIndex = (int)index;
  B[0]         = fMagVector[intIndex].GetBr();
  B[1]         = fMagVector[intIndex].GetBphi();
  B[2]         = fMagVector[intIndex].GetBz();
}

// VcFloat Backend method
template <>
INLINE_CHOICE void MagFieldCellVersion::Gather<vecgeom::kVcFloat>(const typename vecgeom::kVcFloat::precision_v index,
                                                                  typename vecgeom::kVcFloat::precision_v B[3])
{
  typedef typename vecgeom::kVcFloat::Int_t Int_v;
  Int_v ind = (Int_v)index;

  B[0].gather(fMagVector, &MagVector::Br, ind);
  B[1].gather(fMagVector, &MagVector::Bphi, ind);
  B[2].gather(fMagVector, &MagVector::Bz, ind);
}

// Scalar Backend method
template <class Backend>
INLINE_CHOICE void MagFieldCellVersion::NewGather(const typename Backend::precision_v index,
                                                  typename Backend::precision_v B1[3],
                                                  typename Backend::precision_v B2[3],
                                                  typename Backend::precision_v B3[3],
                                                  typename Backend::precision_v B4[3])
{
  // int intIndex= (int) index;

  // //snippet 1
  // B1[0] = fMagCellStructs[intIndex].m0.Br;
  // B1[1] = fMagCellStructs[intIndex].m0.Bphi;
  // B1[2] = fMagCellStructs[intIndex].m0.Bz;

  // B2[0] = fMagCellStructs[intIndex].m1.Br;
  // B2[1] = fMagCellStructs[intIndex].m1.Bphi;
  // B2[2] = fMagCellStructs[intIndex].m1.Bz;

  // B3[0] = fMagCellStructs[intIndex].m2.Br;
  // B3[1] = fMagCellStructs[intIndex].m2.Bphi;
  // B3[2] = fMagCellStructs[intIndex].m2.Bz;

  // B4[0] = fMagCellStructs[intIndex].m3.Br;
  // B4[1] = fMagCellStructs[intIndex].m3.Bphi;
  // B4[2] = fMagCellStructs[intIndex].m3.Bz;

  // snippet 2
  // This thing for scalar
  // because for unknown(yet) reasons, this is faster than snippet 1
  typedef typename Backend::precision_v Float_v;
  Float_v i2 = index + 1;
  Float_v i3 = index + kNoZValues;
  Float_v i4 = i3 + 1;

  Gather<Backend>(index, B1);
  Gather<Backend>(i2, B2);
  Gather<Backend>(i3, B3);
  Gather<Backend>(i4, B4);
}

// VcFloat Backend method
template <>
INLINE_CHOICE void MagFieldCellVersion::NewGather<vecgeom::kVcFloat>(
    const typename vecgeom::kVcFloat::precision_v index, typename vecgeom::kVcFloat::precision_v B1[3],
    typename vecgeom::kVcFloat::precision_v B2[3], typename vecgeom::kVcFloat::precision_v B3[3],
    typename vecgeom::kVcFloat::precision_v B4[3])
{
  typedef typename vecgeom::kVcFloat::Int_t Int_v;
  Int_v ind = (Int_v)index;

  B1[0].gather(fMagCellStructs, &MagCellStructs::m0, &MagVector::Br, ind);
  B1[1].gather(fMagCellStructs, &MagCellStructs::m0, &MagVector::Bphi, ind);
  B1[2].gather(fMagCellStructs, &MagCellStructs::m0, &MagVector::Bz, ind);

  B2[0].gather(fMagCellStructs, &MagCellStructs::m1, &MagVector::Br, ind);
  B2[1].gather(fMagCellStructs, &MagCellStructs::m1, &MagVector::Bphi, ind);
  B2[2].gather(fMagCellStructs, &MagCellStructs::m1, &MagVector::Bz, ind);

  B3[0].gather(fMagCellStructs, &MagCellStructs::m2, &MagVector::Br, ind);
  B3[1].gather(fMagCellStructs, &MagCellStructs::m2, &MagVector::Bphi, ind);
  B3[2].gather(fMagCellStructs, &MagCellStructs::m2, &MagVector::Bz, ind);

  B4[0].gather(fMagCellStructs, &MagCellStructs::m3, &MagVector::Br, ind);
  B4[1].gather(fMagCellStructs, &MagCellStructs::m3, &MagVector::Bphi, ind);
  B4[2].gather(fMagCellStructs, &MagCellStructs::m3, &MagVector::Bz, ind);
}

template <class Backend>
INLINE_CHOICE void MagFieldCellVersion::GetFieldValueRZ(const typename Backend::precision_v &r,
                                                        const typename Backend::precision_v &Z,
                                                        vecgeom::Vector3D<typename Backend::precision_v> &rzField)
{

  typedef typename Backend::precision_v Float_v;

  // Take care that radius and z for out of limit values take values at end points
  Float_v radius = min(r, kRMax);
  Float_v z      = max(min(Z, kZMax), -kZMax);

  // to make sense of the indices, consider any particular instance e.g. (25,-200)
  Float_v rFloor  = floor(radius * kRDiffInv);
  Float_v rIndLow = rFloor * kNoZValues;
  // Float_v rIndHigh = rIndLow + kNoZValues;

  // if we use z-z0 in place of two loops for Z<0 and Z>0
  // z-z0 = [0,32000]
  // so indices 0 to 160 : total 161 indices for (z-z0)/200
  // i.e. we are saying:
  Float_v zInd = floor((z - kZ0) * kZDiffInv);
  // need i1,i2,i3,i4 for 4 required indices
  Float_v i1 = rIndLow + zInd;

  Float_v zLow       = (zInd - kHalfZValues) * kZDiff; // 80 because it's the middle index in 0 to 160
  Float_v zHigh      = zLow + kZDiff;
  Float_v radiusLow  = rFloor * kRDiff;
  Float_v radiusHigh = radiusLow + kRDiff;

  Float_v a1 = (radiusHigh - radius) * (zHigh - z); // area to be multiplied with i1
  Float_v a2 = (radiusHigh - radius) * (z - zLow);
  Float_v a3 = (radius - radiusLow) * (zHigh - z);
  Float_v a4 = (radius - radiusLow) * (z - zLow);

  // All we need is i1. And we can ask one gather to do everything. Let's make new gather.
  Float_v B1[3], B2[3], B3[3], B4[3];
  NewGather<Backend>(i1, B1, B2, B3, B4);

  Float_v BR   = (B1[0] * a1 + B2[0] * a2 + B3[0] * a3 + B4[0] * a4);
  Float_v BPhi = (B1[1] * a1 + B2[1] * a2 + B3[1] * a3 + B4[1] * a4);
  Float_v BZ   = (B1[2] * a1 + B2[2] * a2 + B3[2] * a3 + B4[2] * a4);

  rzField.x() = BR;
  rzField.y() = BPhi;
  rzField.z() = BZ;
}

template <class Backend>
INLINE_CHOICE
    //__attribute__ ((noinline))
    // Sidenote: For theta =0; xyzField = rzField.
    // theta =0 corresponds to y=0

    void
    MagFieldCellVersion::GetFieldValue(const vecgeom::Vector3D<typename Backend::precision_v> &pos,
                                       vecgeom::Vector3D<typename Backend::precision_v> &xyzField)
{

  typedef typename Backend::precision_v Float_v;
  typedef typename Backend::bool_v Bool_v;

  Float_v cyl[2];
  CartesianToCylindrical<Backend>(pos, cyl);
  vecgeom::Vector3D<Float_v> rzField;
  GetFieldValueRZ<Backend>(cyl[0], cyl[1], rzField); // cyl[2] =[r,z]

  // Did Float_v instead of dataType in order to generalize later
  Float_v zero(0.);
  Float_v one(1.);
  // dataType zero = 0.;
  // dataType one = 1.;
  Float_v sinTheta(zero), cosTheta(one); // initialize as theta=0
  // To take care of r =0 case

  // MaskedAssign(cond, value , var );
  // where cond is Bool_v, value is value calculated, var is the variable taking value
  Bool_v nonZero = (cyl[0] != zero);
  Float_v rInv   = zero;
  vecgeom::MaskedAssign<dataType>(nonZero, 1 / cyl[0], &rInv);
  sinTheta = pos.y() * rInv;
  vecgeom::MaskedAssign<dataType>(nonZero, pos.x() * rInv, &cosTheta);

  CylindricalToCartesian<Backend>(rzField, sinTheta, cosTheta, xyzField);
}

#endif
