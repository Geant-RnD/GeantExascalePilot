//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
//
/**
 * @file Geant/proxy/ProxyPhysicsTable.cu
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#include <fstream>
#include <iomanip>

#include "Geant/proxy/ProxyPhysicsTable.cuh"

namespace geantx {

GEANT_HOST
ProxyPhysicsTable::ProxyPhysicsTable() 
{
  fTableSize = 0;
  fNumPhysicsVector = 0;
  fPhysicsVectors = nullptr;
}

GEANT_HOST_DEVICE 
ProxyPhysicsTable::~ProxyPhysicsTable() 
{
  for (int i = 0; i < fNumPhysicsVector; ++i)
    if(!fPhysicsVectors[i]) delete fPhysicsVectors[i];
  delete fPhysicsVectors;
}

GEANT_HOST
bool ProxyPhysicsTable::RetrievePhysicsTable(const std::string& fileName)
{
  std::ifstream fIn;  
  // open input file

  fIn.open(fileName,std::ios::in|std::ios::binary); 

  // check if the file has been opened successfully 
  if (!fIn)
  {
    fIn.close();
    return false;
  }

  // Number of vectors
  fIn >> fNumPhysicsVector;

  fPhysicsVectors = new ProxyPhysicsVector * [fNumPhysicsVector];

  // Physics Vector
  for (size_t idx=0; idx < fNumPhysicsVector; ++idx)
  {
    int vType=0;
    fIn >>  vType;

    fPhysicsVectors[idx] = new ProxyPhysicsVector();

    if (! (fPhysicsVectors[idx]->Retrieve(fIn)) )
    {
      fIn.close();
      return false;
    }
    fPhysicsVectors[idx]->SetType(vType);
    fPhysicsVectors[idx]->SetSpline(true);
  } 
  fIn.close();

  //calcuate the size of this table
  int totalTableSize = 0;
  for (int i = 0; i < fNumPhysicsVector; i++) totalTableSize += fPhysicsVectors[i]->SizeOfVector();
  fTableSize = totalTableSize;

  //fill drived values
  
  return true;
}

#ifdef GEANT_CUDA
void ProxyPhysicsTable::Relocate(void *devPtr)
{
  // device pointers in device memory
  ProxyPhysicsVector **fProxyPhysicsVector_d;
  cudaMalloc((void **)&fProxyPhysicsVector_d, fNumPhysicsVector*fPhysicsVectors[0]->SizeOfVector());

  // device pointers in host memory
  ProxyPhysicsVector* temp_d[fNumPhysicsVector];

  // relocate pointers of this to the corresponding device pointers
  for (int i = 0; i < fNumPhysicsVector; ++i) {
    cudaMalloc((void **)&temp_d[i], fPhysicsVectors[i]->SizeOfVector());
    fPhysicsVectors[i]->Relocate(temp_d[i]);
  }

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(fProxyPhysicsVector_d, temp_d, fTableSize, cudaMemcpyHostToDevice);

  fPhysicsVectors = fProxyPhysicsVector_d;

  // copy the manager from host to device.
  cudaMemcpy(devPtr, this, fTableSize, cudaMemcpyHostToDevice);
}
#endif

GEANT_HOST_DEVICE
double ProxyPhysicsTable::Value(int index, double energy) 
{
  return fPhysicsVectors[index]->Value(energy);
}
  
GEANT_HOST_DEVICE
void ProxyPhysicsTable::Print() 
{
  printf("%d\n",fNumPhysicsVector);
  // check data
  /*
  for(int idx=0; idx < fNumPhysicsVector ; ++idx){
    printf("%d\n", fPhysicsVectors[idx]->fType);
    printf("%f %f %d\n", fPhysicsVectors[idx]->fEdgeMin, fPhysicsVectors[idx]->fEdgeMax, fPhysicsVectors[idx]->fNumberOfNodes);
    printf("%d\n", fPhysicsVectors[idx]->fNumberOfNodes);
    for(int j=0; j< fPhysicsVectors[idx]->fNumberOfNodes; ++j) {
      printf("%f %e\n",fPhysicsVectors[idx]->fBinVector[j], fPhysicsVectors[idx]->fDataVector[j]);
    }
  }
  */
}

} // namespace geantx
