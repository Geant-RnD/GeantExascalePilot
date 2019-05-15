#include "base/Map.h"
#include "base/RNG.h"
#include <cstdlib>
#include <map>
#include <vector>
using vecgeom::RNG;
#include "backend/cuda/Interface.h"

class MyClass {
private:
  float fData;

public:
  VECCORE_ATT_HOST_DEVICE
  MyClass() { fData = 0; }
  VECCORE_ATT_HOST_DEVICE
  MyClass(float data) { fData = data; }
  VECCORE_ATT_HOST_DEVICE
  float getData() const { return fData; }
};

void launchTestNew(vecgeom::DevicePtr<vecgeom::cuda::map<double, MyClass>> &devMap,
                   vecgeom::DevicePtr<double> key, int N, int nBlocks, int nThreads);
void launchRebuildMap(vecgeom::DevicePtr<vecgeom::cuda::map<double, MyClass>> &devMap,
                      vecgeom::DevicePtr<double> key, vecgeom::DevicePtr<MyClass> value,
                      int N, int nBlocks, int nThreads);

VECCORE_ATT_HOST
double getRandom()
{
  return RNG::Instance().uniform();
}
/*
VECCORE_ATT_HOST
void testStd(int size, double* keys,MyClass* values) {
   std::map<double,double> stdMap;
   for (int i=0; i < size; ++i) {
      stdMap.insert(std::pair<double,MyClass>(keys[i],values[i]));
   }

   for (int i=0; i < size; ++i)
      printf("From std map= %f and with find
%f\n",stdMap[keys[i]],stdMap.find(keys[i])->second);

}
*/

int main()
{
  const int kSize    = 50;
  double *mapKeys    = new double[kSize];
  MyClass *mapValues = new MyClass[kSize];

  for (int i = 0; i < kSize; ++i) {
    mapValues[i] = MyClass(getRandom());
    mapKeys[i]   = getRandom();
    printf(" vectors %f, %f\n", mapKeys[i], mapValues[i].getData());
  }

  vecgeom::DevicePtr<double> mapKeysDev;
  mapKeysDev.Allocate(kSize);
  if (cudaGetLastError() != cudaSuccess) {
    printf(" ERROR ALLOC KEYS\n");
    return 0;
  }
  vecgeom::DevicePtr<MyClass> mapValuesDev;
  mapValuesDev.Allocate(kSize);
  if (cudaGetLastError() != cudaSuccess) {
    printf(" ERROR ALLOC VALUES\n");
    return 0;
  }
  vecgeom::DevicePtr<vecgeom::cuda::map<double, MyClass>> devMap;
  devMap.Allocate(kSize);
  if (cudaGetLastError() != cudaSuccess) {
    printf(" ERROR ALLOC MAP\n");
    return 0;
  }
  devMap.Construct();

  mapKeysDev.ToDevice(mapKeys, kSize);
  if (cudaSuccess != cudaGetLastError()) {
    printf("ERROR MEMCPY keys\n");
    return 0;
  }
  mapValuesDev.ToDevice(mapValues, kSize);
  if (cudaSuccess != cudaGetLastError()) {
    printf("ERROR MEMCPY values\n");
  }

  printf(" rebuild map\n");
  launchRebuildMap(devMap, mapKeysDev, mapValuesDev, kSize, 1, 1);
  launchTestNew(devMap, mapKeysDev, kSize, 1, 1);

  delete[] mapKeys;
  delete[] mapValues;
  mapKeysDev.Deallocate();
  mapValuesDev.Deallocate();
  return 0;
}
