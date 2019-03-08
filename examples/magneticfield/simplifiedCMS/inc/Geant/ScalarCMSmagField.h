//===--- (CMS)ScalarCMSmagField.h - Geant-V ------------------------------*- C++ -*-===//
//
//                     Geant-V Prototype
//
//===----------------------------------------------------------------------===//
/**
 * @file   (CMS)ScalarCMSmagField.h
 * @brief  Bi-linear interpolation of CMS-like field
 * @author Ananya
 */
//===----------------------------------------------------------------------===//

/*
 *  Details of current version / choices:
 *   - Reordered the way in which Gather was being used:
 *     Gathered elements for i1 and then i3 (i.e. the next r value)
 *     Then i2, then i4.
 *        The idea is to ensure that the different cache lines are accessed early,
 *        so that they are available when the remaining values are needed, without
 *        further waiting.
 *   - Floats used
 *
 *  Note about ordering of memory:
 *         row  (stagerred in memory)
 *          ||
 *          \/    Column ( consecutive in memory )
 *                  ===>
 *          i        i1       i2 (= i1 +1 )
 *
 *         i+1       i3       i4
 */

#ifndef _SCALARCMSMAGFIELD_H__
#define _SCALARCMSMAGFIELD_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cassert>
#include <ctime>

#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/Global.h"

#include "Geant/VVectorField.h"
#include "Units.h"

// Configuration options - to be improved and incorporated in CMakeLists.txt ??
//
// #define INLINE_CHOICE __attribute__ ((noinline))
#define INLINE_CHOICE inline __attribute__((always_inline))
// #define INLINE_CHOICE inline

template <typename dataType>
struct MagVector3 {
public:
  dataType Br   = 0.;
  dataType Bphi = 0.;
  dataType Bz   = 0.;

public:
  void SetBr(dataType a) { Br = a; }
  void SetBphi(dataType a) { Bphi = a; }
  void SetBz(dataType a) { Bz = a; }
  dataType GetBr() { return Br; }
  dataType GetBphi() { return Bphi; }
  dataType GetBz() { return Bz; }
};

class ScalarCMSmagField : public GUVMagneticField {
public:
  ScalarCMSmagField();
  ScalarCMSmagField(std::string inputMap);
  ScalarCMSmagField(const ScalarCMSmagField &right);

  // Takes as input x,y,z; Gives output Bx,By,Bz
  //
  void GetFieldValue(const vecgeom::Vector3D<double> &pos, vecgeom::Vector3D<double> &xyzField);

  void GetFieldValue(const vecgeom::Vector3D<double> &pos, vecgeom::Vector3D<float> &xyzField) override final;

  // Reads data from given 2D magnetic field map. Can be easily modified to read a given 2D map, in case the file
  // changes
  bool ReadVectorData(std::string inputMap);
  // Return value: success of finding and reading file.

  ~ScalarCMSmagField()
  {
    if (fPrimary) delete[] fMagvArray;
  };

public:
  //  Invariants -- parameters of the field
  const float millimeter = 1.0; // Currently -- to be native GeantV

  const double kRMax     = 9000. * millimeter;  //  Maximum value of R =  9.00 meters
  const double kZMax     = 16000. * millimeter; //  Max value of Z = 16.00 meters
  const int kNoZValues   = 161;
  const int kNoRValues   = 181;
  const int kHalfZValues = 80;

  // Derived values
  // kRDiff and kZDiff take care of mm because they come from kRMax and kZMax which have mm in them
  const float kRDiff = kRMax / (kNoRValues - 1);     //  Radius increment between lattice points
  const float kZDiff = 2 * kZMax / (kNoZValues - 1); //  Z increment

  const float kZ0       = -kZMax;
  const float kRDiffInv = 1.0 / kRDiff;
  const float kZDiffInv = 1.0 / kZDiff;
  const float kAInverse = 1.0 / (kRDiff * kZDiff);

  // For (R,Z) pairs : gives field in cylindrical coordinates in rzfield
  //
  void GetFieldValueRZ(const double &radius, const double &z, vecgeom::Vector3D<double> &rzField);

protected:
  // Used to convert cartesian coordinates to cylindrical coordinates R-Z-phi
  // Does not calculate phi
  //
  void CartesianToCylindrical(const vecgeom::Vector3D<double> &cart, double cyl[2]);

  // Converts cylindrical magnetic field to field in cartesian coordinates
  //
  void CylindricalToCartesian(const vecgeom::Vector3D<double> &rzField, const double sinTheta, const double cosTheta,
                              vecgeom::Vector3D<double> &xyzField);

  void Gather2(const double index, double B1[3], double B2[3]);

public:
  // Methods for Multi-treading
  ScalarCMSmagField *CloneOrSafeSelf(bool *pSafe);
  VVectorField *Clone() const override;

private:
  MagVector3<float> *fMagvArray; //  = new MagVector3<float>[30000];
  bool fReadData;
  bool fVerbose;
  bool fPrimary; /** Read in and own the data arrays */
};

ScalarCMSmagField::ScalarCMSmagField() : fReadData(false), fVerbose(true), fPrimary(false)
{
  fMagvArray    = new MagVector3<float>[kNoZValues * kNoRValues];
  fVcMagVector3 = new Vc::vector<MagVector3<float>>;
  if (fVerbose) {
    printf("%s", "- ScalarCMSmagField class: Version: Reorder2 (floats)");
  }
}

ScalarCMSmagField::ScalarCMSmagField(std::string inputMap) : ScalarCMSmagField()
{
  fMagvArray = new MagVector3<float>[kNoZValues * kNoRValues];

  std::cout << "- ScalarCMSmagField c-tor #2" << std::endl;
  // std::cout<<" Version: Reorder2 (floats) (with VC_NO_MEMBER_GATHER enabled if required)"<<std::endl;
  fReadData = ScalarCMSmagField::ReadVectorData(inputMap);
  fPrimary  = true; // Own the data!
}

ScalarCMSmagField::ScalarCMSmagField(const ScalarCMSmagField &right)
    : fReadData(right.fReadData), fVerbose(right.fVerbose), fPrimary(false)
{
  fMagvArray = right.fMagvArray;

  fVcMagVector3 = right.fVcMagVector3;
}

// ScalarCMSmagField::~ScalarCMSmagField()

// INLINE_CHOICE
bool ScalarCMSmagField::ReadVectorData(std::string inputMap)
{
  std::cout << "- ScalarCMSmagField::ReadVectorData called with filename= " << inputMap << std::endl;
  // printf( "- ScalarCMSmagField::ReadVectorData called with filename= %s\n", inputMap );
  std::string line;
  std::string s1, s2, s3, s4, s5, s0;
  float d1, d2, d3, d4, d5, d0;
  int ind = 0;
  std::ifstream pFile(inputMap);
  if (pFile.is_open()) {
    // getline() returns the stream. testing the stream with while returns error such as EOF
    while (getline(pFile, line)) {
      // so here we know that the read was a success and that line has valid data
      std::stringstream ss(line);
      // parsing all the parts. s0's store the string names which are of no use to us.
      ss >> s0 >> d1 >> s1 >> d0 >> s2 >> d2 >> s3 >> d3 >> s4 >> d4 >> s5 >> d5;

      fMagvArray[ind].SetBr(d4 * kAInverse);
      fMagvArray[ind].SetBphi(d5 * kAInverse);
      fMagvArray[ind].SetBz(d3 * kAInverse);
#if VERBOSE
      if (ind % 10 == 0)
        std::cout << "Read in line " << ind << " Values= " << d3 << " " << d4 << " " << d5 << std::endl;
#endif
      ind++;
    }
    pFile.close();
  } else {
    std::cerr << "Unable to open file (for CMS mag field). Name = '" << inputMap << "'" << std::endl;
    exit(1);
  }
  return true;
}

INLINE_CHOICE
void ScalarCMSmagField::CartesianToCylindrical(const vecgeom::Vector3D<double> &cart, double cyl[2])
{

  // cyl[] =[r,z]
  cyl[0] = sqrt(cart.x() * cart.x() + cart.y() * cart.y()); // r = sqrt(x^2 + y^2)
  cyl[1] = cart.z();                                        // z = z
}

INLINE_CHOICE
void ScalarCMSmagField::CylindricalToCartesian(const vecgeom::Vector3D<double> &rzField, const double sinTheta,
                                               const double cosTheta, vecgeom::Vector3D<double> &xyzField)
{
  // rzField[] has r, phi and z

  xyzField.x() = rzField.x() * cosTheta - rzField.y() * sinTheta; // Bx= Br cos(theta) - Bphi sin(theta)
  xyzField.y() = rzField.x() * sinTheta + rzField.y() * cosTheta; // By = Br sin(theta) + Bphi cos(theta)
  xyzField.z() = rzField.z();                                     // Bz = Bz
}

INLINE_CHOICE
void ScalarCMSmagField::Gather2(const double index, double B1[3], double B2[3])
{

  int intIndex  = (int)index;
  int intIndex2 = intIndex + kNoZValues;

  // Fetch one component of each point first, then the rest.
  B1[0] = fMagvArray[intIndex].GetBr();
  B2[0] = fMagvArray[intIndex2].GetBr();

  B1[1] = fMagvArray[intIndex].GetBphi();
  B1[2] = fMagvArray[intIndex].GetBz();

  B2[1] = fMagvArray[intIndex2].GetBphi();
  B2[2] = fMagvArray[intIndex2].GetBz();
}

INLINE_CHOICE
void ScalarCMSmagField::GetFieldValueRZ(const double &r, const double &Z, vecgeom::Vector3D<double> &rzField)
{

  typedef double Float_v;

  // Take care that radius and z for out of limit values take values at end points
  Float_v radius = std::min(r, kRMax);
  Float_v z      = std::max(std::min(Z, kZMax), -kZMax);

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
  Float_v i2 = i1 + 1;

  Float_v zLow       = (zInd - kHalfZValues) * kZDiff; // 80 because it's the middle index in 0 to 160
  Float_v zHigh      = zLow + kZDiff;
  Float_v radiusLow  = rFloor * kRDiff;
  Float_v radiusHigh = radiusLow + kRDiff;

  Float_v a1 = (radiusHigh - radius) * (zHigh - z); // area to be multiplied with i1
  Float_v a2 = (radiusHigh - radius) * (z - zLow);
  Float_v a3 = (radius - radiusLow) * (zHigh - z);
  Float_v a4 = (radius - radiusLow) * (z - zLow);

  Float_v B1[3], B2[3], B3[3], B4[3];
  Gather2(i1, B1, B3);
  Gather2(i2, B2, B4);

  Float_v BR   = B1[0] * a1 + B2[0] * a2 + B3[0] * a3 + B4[0] * a4;
  Float_v BPhi = B1[1] * a1 + B2[1] * a2 + B3[1] * a3 + B4[1] * a4;
  Float_v BZ   = B1[2] * a1 + B2[2] * a2 + B3[2] * a3 + B4[2] * a4;

  rzField.x() = BR;
  rzField.y() = BPhi;
  rzField.z() = BZ;
}

INLINE_CHOICE
// __attribute__ ((noinline))

void ScalarCMSmagField::GetFieldValue(const vecgeom::Vector3D<double> &pos, vecgeom::Vector3D<double> &xyzField)
{
  //  Sidenote: For theta =0; xyzField = rzField.
  //  theta =0 corresponds to y=0

  using Float_v = double;
  using Bool_v  = bool;

  Float_v cyl[2];
  CartesianToCylindrical(pos, cyl);
  vecgeom::Vector3D<Float_v> rzField;
  GetFieldValueRZ(cyl[0], cyl[1], rzField); // cyl[2] =[r,z]

  float zero = 0.0f;
  float one  = 1.0f;
  Float_v sinTheta(zero), cosTheta(one); // initialize as theta=0
  // To take care of r =0 case

  // MaskedAssign(cond, value , var );
  // where cond is Bool_v, value is value calculated, var is the variable taking value

  bool xNonZero = (cyl[0] != zero);
  if (xNonZero) {
    rInv = 1.0f / cyl[0];
  }
  // Float_v rInv   = zero;
  // vecgeom::MaskedAssign(xNonZero, 1.0f/cyl[0]    , &rInv    );

  sinTheta = pos.y() * rInv;
  // vecgeom::MaskedAssign(xNonZero, pos.x()*rInv, &cosTheta);
  if (xNonZero) cosTheta = pos.x() * rInv;

  CylindricalToCartesian(rzField, sinTheta, cosTheta, xyzField);

  xyzField *= fieldUnits::tesla;

  // std::cout<< "Input pos is: " << pos << " , xyzField is: " << xyzField << std::endl;
}

void ScalarCMSmagField::GetFieldValue(const vecgeom::Vector3D<double> &pos_d, vecgeom::Vector3D<float> &xyzField)
{
  // Call the method
  //    GetFieldValue(const vecgeom::Vector3D<float>      &pos,
  //                        vecgeom::Vector3D<float> &xyzField)

  vecgeom::Vector3D<double> xyzField_d;
  GetFieldValue(pos_d, xyzField_d);

  xyzField = xyzField_d;
}

// This class is thread safe.  So other threads can use the same instance
//

ScalarCMSmagField *ScalarCMSmagField::CloneOrSafeSelf(bool *pSafe)
{
  if (pSafe) *pSafe = true;
  return this;
}

VVectorField *ScalarCMSmagField::Clone() const
{
  return new ScalarCMSmagField(*this);
}

#undef NO_INLINE
#undef FORCE_INLINE
#undef INLINE_CHOICE

#endif
