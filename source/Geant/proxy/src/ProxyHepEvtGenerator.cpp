//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file ProxyHepEvtGenerator.cpp
 * @brief ProxyHepEvtGeneratores
 */
//===----------------------------------------------------------------------===//

#include "Geant/proxy/ProxyHepEvtGenerator.hpp"

namespace geantx {

ProxyHepEvtGenerator::ProxyHepEvtGenerator(const char* evtfile) 
: ProxyEventGenerator<ProxyHepEvtGenerator>()
{
  fInputFile.open((char*)evtfile);

  if (fInputFile.is_open()) {
    fFileName = evtfile;
    std::cout << "ProxyHepEvtGenerator: input " << fFileName << " is open." << std::endl;
  }
  else {
    std::cout << "ProxyHepEvtGenerator: cannot open " << fFileName  << std::endl;
  }
}

ProxyHepEvtGenerator::ProxyHepEvtGenerator(int nevts, const char* evtfile) 
  : ProxyEventGenerator<ProxyHepEvtGenerator>(nevts)
{

  fInputFile.open((char*)evtfile);

  if (fInputFile.is_open()) {
    fFileName = evtfile;
    std::cout << "ProxyHepEvtGenerator: input " << fFileName << " is open." << std::endl;
  }
  else {
    std::cout << "ProxyHepEvtGenerator: cannot open " << fFileName  << std::endl;
  }
}

ProxyEvent* ProxyHepEvtGenerator::GenerateOneEvent()
{ 
  // G4HEPEvtInterface::GeneratePrimaryVertex(G4Event* evt)

  int NHEP = 0;  // number of entries
  if (fInputFile.is_open()) {
    fInputFile >> NHEP;
    std::cout << " NHEP =  " <<  NHEP << std::endl;
  }
  else {
    std::cout << " cannot open file " << std::endl;
    exit(-1);
  }

  if( fInputFile.eof() )  {
    std::cout << " End-Of-File : HEPEvt input file " << std::endl;
  }

  HepEvtData *hepg = new HepEvtData[NHEP];

  for( int i = 0 ; i < NHEP ; ++i) {

    fInputFile >> hepg[i].ISTHEP >> hepg[i].IDHEP >> hepg[i].JDAHEP1 >> hepg[i].JDAHEP2
      	       >> hepg[i].PHEP1  >> hepg[i].PHEP2 >> hepg[i].PHEP3   >> hepg[i].PHEP4;

    if( fInputFile.eof() ) {
      std::cout << " Unexpected End-Of-File : HEPEvt input file" << std::endl;
      exit(-1);
    }
  }

  //count the number of stable particles
  int numParticles = 0;

  for(int i = 0 ; i < NHEP; ++i) {
    if( hepg[i].JDAHEP1 > 0 ) { //  it has daughters
      int jda1 = hepg[i].JDAHEP1-1;
      int jda2 = hepg[i].JDAHEP2-1;

      for( int j = jda1 ; j <= jda2; ++j)  {
        if(hepg[j].ISTHEP>0) {
      	  hepg[j].ISTHEP = -1;
          if( hepg[j].JDAHEP1 == 0) ++numParticles;
      	}
	//else : initial particles
      }
    }
  }

  std::cout << "numParticles = " <<  numParticles << std::endl;

  //proxy: ignore secondary vertices and set stable particles to the primary vertex
  //TODO: implement something similar with G4PrimaryParticle -> G4PrimaryTransformer -> G4Track

  ProxyVertex *primaryVertex = new ProxyVertex();

  for( int i = 0; i < NHEP; ++i) {
    if( (hepg[i].ISTHEP == -1) &&  (hepg[i].JDAHEP1 == 0) ) { 
      TrackState* atrack = new TrackState();
      atrack->fPhysicsState.fMomentum =  hepg[i].PHEP1;
      atrack->fSchedulingState.fGVcode =  hepg[i].IDHEP;
      primaryVertex->AddParticle(atrack);
    }
  }

  ProxyEvent *anEvent = new ProxyEvent();
  anEvent->AddVertex(primaryVertex);

  //clean up
  delete hepg;

  return anEvent;
}

} // namespace geantx
