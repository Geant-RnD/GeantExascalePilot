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
 * @file Geant/proxy/ProxyDataManager.cu
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#include <iostream>
#include <fstream>
#include <iomanip>

#include "Geant/proxy/ProxyDataManager.cuh"
#include "Geant/proxy/ProxyPhysicsTableIndex.hpp"

namespace geantx {

ProxyDataManager *ProxyDataManager::fInstance = 0;

GEANT_HOST
ProxyDataManager *ProxyDataManager::Instance()
{
  if (fInstance == 0) fInstance = new ProxyDataManager();
  return fInstance;
}

GEANT_HOST
ProxyDataManager::ProxyDataManager() 
{
  fSizeOfObject = 2*sizeof(int);
  fNumPhysicsTables = ProxyPhysicsTableIndex::kNumberPhysicsTable;
  fPhysicsTables = new ProxyPhysicsTable * [fNumPhysicsTables];

  RetrievePhysicsData();
  RetrieveCutsTable();
}

GEANT_HOST_DEVICE 
ProxyDataManager::~ProxyDataManager() 
{
  for (int i = 0; i < fNumPhysicsTables ; ++i)
    if(!fPhysicsTables[i]) delete fPhysicsTables[i];
  delete fPhysicsTables;

  delete [] fCutsTable;
}

GEANT_HOST
bool ProxyDataManager::RetrievePhysicsData(/* const std::string& dir */)
{
  //retrieve physics tables (from the given directory)

  char filename[256];

  for(int it = 0 ; it < ProxyPhysicsTableIndex::kNumberPhysicsTable ; ++it) {

    sprintf(filename,"data/table/%s",ProxyPhysicsTableName[it].c_str());

    ++fNumPhysicsTables;
    fPhysicsTables[it] = new ProxyPhysicsTable();

    bool status = fPhysicsTables[it]->RetrievePhysicsTable(filename);
    if(status) {
      size_t size = fPhysicsTables[it]->SizeOfTable();
      std::cout << "Retrieved " << filename << " Size = " << size << std::endl;
      int nvector = fPhysicsTables[it]->NumberOfVector();

      fSizeOfObject += size;
      //TODO: set spline
      for(int iv = 0 ; iv < nvector ; ++iv) {
	// (fPhysicsTables[it]->GetVector())[iv]->SetSpline(useSpline);
      }
    }  
    else {
      std::cout << "Failed to retrieve " << filename << std::endl;
    }
  }

  return true;
}

GEANT_HOST
bool ProxyDataManager::RetrieveCutsTable(/* const std::string& dir */)
{
  //retrieve material cut tables (from the given directory)

  char fileName[256];
  sprintf(fileName,"data/table/%s","cut.dat");

  std::ifstream fIn;  
  // open input file
  fIn.open(fileName,std::ios::in|std::ios::binary); 
  // check if the file has been opened successfully 
  if (!fIn)
  {
    fIn.close();
    return false;
  }

  // Number of materials
  std::string version;

  fIn >> version;
  fIn >> fNumOfCuts;

  double cutLength;
  double cutEnergy;

  fCutsTable = new double [data::nParticleForCuts*fNumOfCuts];

  for (size_t idx=0; idx < fNumOfCuts ; ++idx) {
    for (size_t ipart=0; ipart < data::nParticleForCuts ; ++ipart) {
      fIn >> cutLength >> cutEnergy;
      fCutsTable[data::nParticleForCuts*idx+ipart] = cutEnergy; // [idx=*4 + i]
    }
  }

  return true;
}

#ifdef GEANT_CUDA
void ProxyDataManager::RelocatePhysicsData(void *devPtr)
{
  // device pointers in device memory
  ProxyPhysicsTable **fProxyPhysicsTables_d;

  int totalTableSize = 0;
  for (int it = 0; it < fNumPhysicsTables; it++) totalTableSize += fPhysicsTables[it]->SizeOfTable();

  cudaMalloc((void **)&fProxyPhysicsTables_d, sizeof(totalTableSize));

  // device pointers in host memory
  ProxyPhysicsTable *tables_d[fNumPhysicsTables];

  // relocate pointers of this to the corresponding device pointers
  for (int i = 0; i < fNumPhysicsTables; ++i) {
    cudaMalloc((void **)&tables_d[i], fPhysicsTables[i]->SizeOfTable());
    fPhysicsTables[i]->Relocate(tables_d[i]);
  }

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(fProxyPhysicsTables_d, tables_d, totalTableSize, cudaMemcpyHostToDevice);
  fPhysicsTables = fProxyPhysicsTables_d;

  // copy the manager from host to device.
  cudaMemcpy(devPtr, this, totalTableSize, cudaMemcpyHostToDevice);
}
#endif


GEANT_HOST_DEVICE
void ProxyDataManager::Print() 
{
  printf("%d\n",fNumPhysicsTables);
}

GEANT_HOST_DEVICE
void ProxyDataManager::PrintCutsTable() 
{
  printf("fNumOfCuts=   %d\n",fNumOfCuts);
  for (size_t idx=0; idx < fNumOfCuts ; ++idx) {
    for (size_t ipart=0; ipart < data::nParticleForCuts ; ++ipart) {
      printf("   %f\n",fCutsTable[data::nParticleForCuts*idx+ipart]);
    }
  }

}


} // namespace geantx
