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
 * @file Geant/proxy/NistElementManager.cu
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/NistElementManager.cuh"

#include <fstream>
#include <iomanip>

namespace geantx {

NistElementManager *NistElementManager::fInstance = 0;

GEANT_HOST
NistElementManager *NistElementManager::Instance()
{
  if (fInstance == 0) fInstance = new NistElementManager();
  return fInstance;
}

GEANT_HOST
NistElementManager::NistElementManager()  
  : fNumberOfElements(-1), fNumberOfIsotopes(-1), 
     fElement(nullptr), fIsotope(nullptr)
{
  bool pass = LoadData();
  if(pass) {
    fElement[0].fFirstIsotopeIndex = 0;    
    for(int i = 1 ; i < fNumberOfElements ; ++i) {
      fElement[i].fFirstIsotopeIndex = fElement[i-1].fFirstIsotopeIndex     
                                     + fElement[i-1].fNumberOfIsotopes;      
    }

    for(int i = 0 ; i < fNumberOfElements ; ++i) {
      double weight = 0.0;
      int first = fElement[i].fFirstIsotopeIndex;
      int last  = first + fElement[i].fNumberOfIsotopes;
      for(int j = first ; j < last ; ++j) {
        weight += fIsotope[j].fAtomicMass*fIsotope[j].fComposition; 
      }
      fElement[i].fAtomicWeight = weight;
    }
  }
}

GEANT_HOST
bool NistElementManager::CheckZ(int Z) const
{
   if(Z < 1 || Z > fNumberOfElements) {
     std::cout << "Invalid Z = " << Z << std::endl;
     exit(0);
   }
   return true;
}

GEANT_HOST
bool NistElementManager::CheckZN(int Z, int N) const
{
  if( CheckZ(Z) ) {
    int first = fElement[Z-1].fFirstMassNumber;
    int last = first + fElement[Z-1].fNumberOfIsotopes - 1;
    if(N < first || N > last) {
      std::cout << "Invalid N = " << N << " for the given Z = " << Z << std::endl;
      exit(0);
    }
  }
  return true;
}

GEANT_HOST
std::string NistElementManager::GetElementName(int Z) const
{
  CheckZ(Z);
  return fElement[Z-1].fName;
}

GEANT_HOST
int NistElementManager::GetNumberOfIsotopes(int Z) const
{
  CheckZ(Z);
  return fElement[Z-1].fNumberOfIsotopes;
}

GEANT_HOST
int NistElementManager::GetFirstMassNumber(int Z) const
{
  CheckZ(Z);
  return fElement[Z-1].fFirstMassNumber;
}

GEANT_HOST
double NistElementManager::GetStandardAtomicWeight(int Z) const
{
  CheckZ(Z);
  return fElement[Z-1].fAtomicWeight;
}

GEANT_HOST
double NistElementManager::GetRelativeAtomicMass(int Z, int N) const
{
  int index = GetIsotopeIndex(Z, N);
  return fIsotope[index].fAtomicMass;
}

GEANT_HOST
double NistElementManager::GetIsotopicComposition(int Z, int N) const
{
  int index = GetIsotopeIndex(Z, N);
  return fIsotope[index].fComposition;
}

GEANT_HOST
int NistElementManager::GetIsotopeIndex(int Z, int N) const
{
   CheckZN(Z, N);

   int offset = 0;
   int first = fElement[Z-1].fFirstMassNumber;
   int next  = first + fElement[Z-1].fNumberOfIsotopes;

   for(int i = first ; i < next ; ++i) {
     if(i < N) {
       ++offset;
       continue;
     }
   }

   return fElement[Z-1].fFirstIsotopeIndex + offset;
}

GEANT_HOST
void NistElementManager::Print(int Z) 
{
  CheckZ(Z);
  
  std::cout << "Element Z = " << Z << " " 
    << fElement[Z-1].fName  << " "
    << fElement[Z-1].fNumberOfIsotopes  << " "
    << fElement[Z-1].fAtomicWeight  << " "
    << std::endl;

  int first = fElement[Z-1].fFirstIsotopeIndex;
  int last  = first + fElement[Z-1].fNumberOfIsotopes;

  int cnt = fElement[Z-1].fFirstMassNumber;

  for(int i = first ; i < last ; ++i) {
    std::cout << " N = " << cnt << " " << fIsotope[i].fAtomicMass  << " " 
              << fIsotope[i].fComposition << " " << std::endl;
    ++cnt;
  }
}

GEANT_HOST
bool NistElementManager::LoadData()
{
  //retrieve elements information

  char fileName[256];
  sprintf(fileName,"data/nist/%s","elements.dat");

  std::ifstream fIn;  
  fIn.open(fileName,std::ios::in|std::ios::binary); 

  if (!fIn)
  {
    fIn.close();
    return false;
  }

  fIn >> fNumberOfElements >> fNumberOfIsotopes;

  fElement = new NistElement[fNumberOfElements];
  fIsotope = new NistIsotope[fNumberOfIsotopes];

  // read NIST elements and isotopes data

  for (int i = 0; i <  fNumberOfElements ; ++i) {
    fIn >> fElement[i].fName >> fElement[i].fNumberOfIsotopes
        >> fElement[i].fFirstMassNumber;
  }
  for (int i = 0; i <  fNumberOfIsotopes ; ++i) {
    fIn >> fIsotope[i].fAtomicMass >> fIsotope[i].fComposition;
  }

  return true;
}

} // namespace geantx
