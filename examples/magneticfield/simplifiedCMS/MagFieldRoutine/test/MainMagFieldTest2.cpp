#include "iostream"
#include "MagField.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/Global.h"
#include "test/unit_tests/ApproxEqual.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath> //for sqrt

// ensure asserts are compiled in
#undef NDEBUG
#include <cassert>

using namespace std;

class ReadVectorData {
public:
  vector<double> fRadius, fPhi, fZ, fBr, fBz, fBphi;
  ReadVectorData(string inputMap)
  {
    dataFile = inputMap;
    PleaseReadData();
  };

  ~ReadVectorData() {}

private:
  string dataFile;
  void PleaseReadData()
  {
    string line;
    string s1, s2, s3, s4, s5, s0;
    double d1, d2, d3, d4, d5, d0;
    ifstream pFile(dataFile);
    if (pFile.is_open()) {
      while (getline(pFile, line)) {
        stringstream ss(line);
        ss >> s0 >> d1 >> s1 >> d0 >> s2 >> d2 >> s3 >> d3 >> s4 >> d4 >> s5 >> d5;
        fRadius.push_back(d1);
        fPhi.push_back(d0);
        fZ.push_back(d2);
        fBz.push_back(d3);
        fBr.push_back(d4);
        fBphi.push_back(d5);
      }
      pFile.close();
    } else {
      cout << "Unable to open file" << endl;
    }
  }
};

int main()
{

  MagField m1;
  m1.ReadVectorData("examples/magneticfield/simplifiedCMS/cms2015.txt");

  ReadVectorData dataMap("examples/magneticfield/simplifiedCMS/cms2015.txt");

  const double kRDiff     = 50;
  const double kZDiff     = 200;
  const double kRDiffInv  = 1.0 / kRDiff;
  const double kZDiffInv  = 1.0 / kZDiff;
  const double kRMax      = 9000;
  const double kZMax      = 16000;
  const int noZValues     = 161;
  const int halfZValues   = 80;
  const double halfWeight = 0.5;

  //(r,0,z) corresponds exactly to (r,z) in terms that xyzField obtained is same as rzField since
  // theta=0 in this case. Hence can check GetFieldValueXYZ in place of GetFieldValueTest
  // Limitation however is that can't check for points with non zero y.
  for (double r = 0.; r <= kRMax; r = r + kRDiff) {
    for (double z = -kZMax; z <= kZMax; z = z + kZDiff) {
      // Checks for (r,0,z) and (r,0,z) against (r,z)
      vecgeom::Vector3D<double> pos1(r, 0., z);
      vecgeom::Vector3D<double> xyzField1;
      m1.GetFieldValueXYZ(pos1, xyzField1);

      int i = r * kRDiffInv * noZValues + halfZValues + z * kZDiffInv;

      vecgeom::Vector3D<double> rzCheckField1(dataMap.fBr[i], dataMap.fBphi[i], dataMap.fBz[i]);

      assert(ApproxEqual(xyzField1, rzCheckField1));
    }
  }

  // Check for points on mid of cell lines i.e. (r/2,0,z) , (r,0,z/2)
  for (double r = 0.; r < kRMax; r = r + kRDiff) {
    for (double z = -kZMax; z < kZMax; z = z + kZDiff) {
      vecgeom::Vector3D<double> pos2(r + kRDiff * halfWeight, 0., z), pos3(r, 0., z + kZDiff * halfWeight);
      vecgeom::Vector3D<double> xyzField2, xyzField3;
      m1.GetFieldValueXYZ(pos2, xyzField2);
      m1.GetFieldValueXYZ(pos3, xyzField3);

      // Say i1, i2, i3, i4
      vecgeom::Vector3D<double> rzCheckField2, rzCheckField3;
      // Now need i1, i2, i3, i4
      // for pos2 and pos5, take i1 and i2. i4 = i3 + 161. Same z, different r. so skip through as many values of z as
      // for one r
      int i1 = r * kRDiffInv * noZValues + halfZValues + z * kZDiffInv;
      int i2 = i1 + noZValues;

      // for pos3 and pos7, take i3 and i4. Then i4 = i3+1 because same r
      // int i3 = r*kRDiffInv*noZValues + halfZValues + z*kZDiffInv;
      int i3 = i1;
      int i4 = i3 + 1;

      rzCheckField2.x() = (dataMap.fBr[i1] + dataMap.fBr[i2]) * halfWeight;
      rzCheckField2.y() = (dataMap.fBphi[i1] + dataMap.fBphi[i2]) * halfWeight;
      rzCheckField2.z() = (dataMap.fBz[i1] + dataMap.fBz[i2]) * halfWeight;

      rzCheckField3.x() = (dataMap.fBr[i3] + dataMap.fBr[i4]) * halfWeight;
      rzCheckField3.y() = (dataMap.fBphi[i3] + dataMap.fBphi[i4]) * halfWeight;
      rzCheckField3.z() = (dataMap.fBz[i3] + dataMap.fBz[i4]) * halfWeight;

      assert(ApproxEqual(xyzField2, rzCheckField2)); // this works
      assert(ApproxEqual(xyzField3, rzCheckField3));
    }
  }

  // For point in middle of cell
  for (double r = 0.; r < kRMax; r = r + kRDiff) {
    for (double z = -kZMax; z < kZMax; z = z + kZDiff) {
      vecgeom::Vector3D<double> pos4(r + kRDiff * halfWeight, 0., z + kZDiff * halfWeight);
      vecgeom::Vector3D<double> xyzField4, rzCheckField4;
      m1.GetFieldValueXYZ(pos4, xyzField4);

      // need to get rzcheckfield4
      // going to be average of 4 points
      int i1 = r * kRDiffInv * noZValues + halfZValues + z * kZDiffInv;
      int i2 = i1 + noZValues;
      int i3 = i1 + 1;
      int i4 = i2 + 1;

      rzCheckField4.x() = (dataMap.fBr[i1] + dataMap.fBr[i2] + dataMap.fBr[i3] + dataMap.fBr[i4]) * 0.25;
      rzCheckField4.y() = (dataMap.fBphi[i1] + dataMap.fBphi[i2] + dataMap.fBphi[i3] + dataMap.fBphi[i4]) * 0.25;
      rzCheckField4.z() = (dataMap.fBz[i1] + dataMap.fBz[i2] + dataMap.fBz[i3] + dataMap.fBz[i4]) * 0.25;

      assert(ApproxEqual(xyzField4, rzCheckField4)); // this works
    }
  }

  return 0;
}
