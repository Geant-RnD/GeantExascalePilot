//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyEventGenerator.hpp
 * @brief A base of Proxy Event Generator
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyEvent.hpp"

namespace geantx {

template <typename TGenerator> struct Generator_traits;

template <class TGenerator>
class ProxyEventGenerator {

  using EventVector_t = ProxyVector<ProxyEvent*>;

public:

  ProxyEventGenerator() : fNumberOfEvents(0), fEventVector(nullptr) {}

  ProxyEventGenerator(int nevts) : fNumberOfEvents(nevts)
    { fEventVector = new EventVector_t (nevts); }

  ~ProxyEventGenerator() { fEventVector->clear(); }

  // common methods
  int GetNumberOfEvents() const { return fNumberOfEvents; }

  EventVector_t* GetEventVector() const { return fEventVector; }

  void GenerateEvents() { GenerateEvents(fNumberOfEvents); }

  void GenerateEvents(int nevents);

  // mandatory static methods
  ProxyEvent* GenerateOneEvent() 
   { return static_cast<TGenerator *>(this) -> GenerateOneEvent(); }

protected:
  int fNumberOfEvents;
  EventVector_t* fEventVector;
};

template <typename TGenerator>
void ProxyEventGenerator<TGenerator>::GenerateEvents(int nevents) 
{
  for(int i = 0; i < nevents ; ++i) {
    fEventVector->push_back(GenerateOneEvent());
  }
}


} // namespace geantx
