//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxySingleton.hpp
 * @brief the base of proxy singleton classes
 */
//===----------------------------------------------------------------------===//

#pragma once

namespace geantx {

template <typename T>
class ProxySingleton {

protected:
  ProxySingleton() {}

public:

  ProxySingleton(const ProxySingleton &) = delete;
  ProxySingleton& operator= (const ProxySingleton &) = delete;
  ProxySingleton(ProxySingleton &&) = delete;
  ProxySingleton& operator= (ProxySingleton &&) = delete;

  static T* Instance() {
    static T* instance = new T;
    return instance;  
  }

};

} // namespace geantx
