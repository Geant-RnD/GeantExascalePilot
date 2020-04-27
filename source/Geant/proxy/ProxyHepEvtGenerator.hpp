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
 * @file Geant/proxy/ProxyHepEvtGenerator.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include <string>
#include <fstream>

#include "Geant/proxy/ProxyEvent.hpp"
#include "Geant/proxy/ProxyEventGenerator.hpp"

namespace geantx
{

class ProxyHepEvtGenerator;

template <> struct Generator_traits<ProxyHepEvtGenerator>
{
  using Generator_t = ProxyHepEvtGenerator;
};

struct HepEvtData {
  int    ISTHEP;  // status code
  int    IDHEP;   // PDG code
  int    JDAHEP1; // first daughter
  int    JDAHEP2; // last daughter
  double PHEP1;   // px in GeV
  double PHEP2;   // py in GeV
  double PHEP3;   // pz in GeV
  double PHEP4;   // mass in GeV
};

class ProxyHepEvtGenerator : public ProxyEventGenerator<ProxyHepEvtGenerator>
{
  using this_type = ProxyHepEvtGenerator;
  friend class ProxyEventGenerator<ProxyHepEvtGenerator>;
public:
  
  ProxyHepEvtGenerator(const char* evtfile = "hepevt.data");
  ProxyHepEvtGenerator(int nevts, const char* evtfile = "hepevt.data");

  ~ProxyHepEvtGenerator() = default;
  
  // mandatory methods
  ProxyEvent* GenerateOneEvent(); 

private:
  std::string fFileName;
  std::ifstream fInputFile;
};

}  // namespace geantx
