//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/core/Memory.hpp
 * @brief Wrapper(s) around memory pointers
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#if !defined(GEANT_HOST_DEVICE)
#    define GEANT_HOST_DEVICE __host__ __device__
#endif

namespace geantx
{
//----------------------------------------------------------------------------//
//  Simple wrapper to designate as a pointer to memory on device
//
template <typename _Tp>
class DevicePtr
{
public:
    using this_type = DevicePtr<_Tp>;

public:
    GEANT_HOST_DEVICE DevicePtr()
    : fPtr(nullptr)
    {}
    GEANT_HOST_DEVICE DevicePtr(_Tp* ptr)
    : fPtr(ptr)
    {}
    GEANT_HOST_DEVICE ~DevicePtr()                = default;
    GEANT_HOST_DEVICE DevicePtr(const this_type&) = default;
    GEANT_HOST_DEVICE DevicePtr(this_type&&)      = default;

    // operators
    GEANT_HOST_DEVICE explicit operator _Tp*() { return fPtr; }
    GEANT_HOST_DEVICE explicit operator void*() { return fPtr; }
    GEANT_HOST_DEVICE this_type& operator=(const this_type&) = default;
    GEANT_HOST_DEVICE this_type& operator=(this_type&&) = default;

    GEANT_HOST_DEVICE _Tp* get() const { return fPtr; }

private:
    _Tp* fPtr;
};

}  // namespace geantx
