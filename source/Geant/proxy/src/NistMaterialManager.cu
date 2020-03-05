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
 * @file Geant/proxy/NistMaterialManager.cu
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/NistMaterialManager.cuh"

#include <fstream>
#include <iomanip>

namespace geantx {

NistMaterialManager *NistMaterialManager::fInstance = 0;

GEANT_HOST
NistMaterialManager *NistMaterialManager::Instance()
{
  if (fInstance == 0) fInstance = new NistMaterialManager();
  return fInstance;
}

GEANT_HOST
NistMaterialManager::NistMaterialManager()  
  : fNumberOfMaterials(-1), fMaterial(nullptr)
{
  bool pass = LoadData();

  if(pass) {
    //construct NIST materials
  }
}

GEANT_HOST
void NistMaterialManager::PrintMaterialList() 
{
  std::cout << "<--- List of NIST Materials --->" << std::endl;
  std::cout << "  Number of Materials = " << fNumberOfMaterials << std::endl;
  for (int i = 0; i <  fNumberOfMaterials ; ++i) {
    std::cout << fMaterial[i].fName << std::endl;
  }
}

GEANT_HOST
void NistMaterialManager::Print() 
{
  std::cout << std::endl;
  std::cout << "NIST Materials: Number of Materials = " << fNumberOfMaterials << std::endl;
  
  for (int i = 0; i <  fNumberOfMaterials ; ++i) {
    std::cout << fMaterial[i].fName << " " << fMaterial[i].fDensity << " "
        << fMaterial[i].fZ << " " << fMaterial[i].fMeanExcitationEnergy << " " 
        << fMaterial[i].fNumberOfElements << " " << fMaterial[i].fState << std::endl;

    if(fMaterial[i].fNumberOfElements > 1) {
      for (int j = 0; j < fMaterial[i].fNumberOfElements ; ++j) {
        std::cout << (fMaterial[i].fElement)[j].fZ << " " << (fMaterial[i].fElement)[j].fWeight << std::endl;
      }
    }
  }
}

GEANT_HOST
bool NistMaterialManager::LoadData()
{
  //retrieve elements information

  char fileName[256];
  sprintf(fileName,"data/nist/%s","materials.dat");

  std::ifstream fIn;  
  fIn.open(fileName,std::ios::in|std::ios::binary); 

  if (!fIn)
  {
    fIn.close();
    return false;
  }

  fIn >> fNumberOfMaterials; 

  fMaterial = new NistMaterial[fNumberOfMaterials];

  // read NIST elements and isotopes data

  for (int i = 0; i <  fNumberOfMaterials ; ++i) {
    fIn >> fMaterial[i].fName >> fMaterial[i].fDensity
        >> fMaterial[i].fZ >> fMaterial[i].fMeanExcitationEnergy 
        >> fMaterial[i].fNumberOfElements >> fMaterial[i].fState;

    if(fMaterial[i].fNumberOfElements > 1) {
      fMaterial[i].fElement = new NistElementPrimitive [fMaterial[i].fNumberOfElements];
      for (int j = 0; j < fMaterial[i].fNumberOfElements ; ++j) {
        fIn >> (fMaterial[i].fElement)[j].fZ >> (fMaterial[i].fElement)[j].fWeight;
      }
    }
  }

  return true;
}

} // namespace geantx
