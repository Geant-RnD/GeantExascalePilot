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

GEANT_HOST_DEVICE 
ProxyPhysicsTable::ProxyPhysicsTable() 
{
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
  } 
  fIn.close();
  return true;
}

#ifdef GEANT_CUDA
void ProxyPhysicsTable::Relocate(void *devPtr)
{
  // device pointers in device memory
  ProxyPhysicsVector **fProxyPhysicsVector_d;

  int totalTableSize = 0;
  for (int i = 0; i < fNumPhysicsVector; i++) totalTableSize += fPhysicsVectors[i]->SizeOfVector();

  cudaMalloc((void **)&fProxyPhysicsVector_d, sizeof(totalTableSize));

  // device pointers in host memory
  ProxyPhysicsVector *tables_d[fNumPhysicsVector];

  // relocate pointers of this to the corresponding device pointers
  for (int i = 0; i < fNumPhysicsVector; ++i) {
    cudaMalloc((void **)&tables_d[i], fPhysicsVectors[i]->SizeOfVector());
    fPhysicsVectors[i]->Relocate(tables_d[i]);
  }

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(fProxyPhysicsVector_d, tables_d, totalTableSize, cudaMemcpyHostToDevice);
  fPhysicsVectors = fProxyPhysicsVector_d;

  // copy the manager from host to device.
  cudaMemcpy(devPtr, this, totalTableSize, cudaMemcpyHostToDevice);
}
#endif


GEANT_HOST_DEVICE
void ProxyPhysicsTable::Print() 
{
  printf("%d\n",fNumPhysicsVector);
  /*
  // check data
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
