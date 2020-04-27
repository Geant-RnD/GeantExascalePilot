//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyEvent.hpp
 * @brief vertex of primary particles
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyVertex.hpp"

namespace geantx {

class ProxyEvent {

  using Vertex_t = ProxyVertex;
  using VertexVector = ProxyVector<Vertex_t*>;

public:

  ProxyEvent() : fEventId(0) { fVertexVector = new VertexVector(); }

  ~ProxyEvent() { fVertexVector->clear(); }

  int GetEventId() const { return fEventId; }

  int GetNumberOfVertices() const { return fVertexVector->size(); }

  VertexVector* GetVertexVector() const { return fVertexVector; }

  void SetEventId(int evtId) { fEventId = evtId; }
 
  void AddVertex(Vertex_t* vertex) { fVertexVector->push_back(vertex) ; }

private:
  int fEventId;
  VertexVector* fVertexVector;

  //TODO: add hit collection
};

} // namespace geantx
