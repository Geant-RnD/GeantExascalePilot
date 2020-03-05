//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyVector.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx {

namespace Internal {
template <typename T>
struct AllocTrait {

  // Allocate raw buffer to hold the element.
  GEANT_HOST_DEVICE
  static T *Allocate(size_t nElems) { return reinterpret_cast<T *>(new char[nElems * sizeof(T)]); }

  // Release raw buffer to hold the element.
  GEANT_HOST_DEVICE
  static void Deallocate(T *startBuffer) { delete[]((char *)startBuffer); }

  GEANT_HOST_DEVICE
  static void Destroy(T &obj) { obj.~T(); };

  GEANT_HOST_DEVICE
  static void Destroy(T *arr, size_t nElem)
  {
    for (size_t i = 0; i < nElem; ++i)
      Destroy(arr[i]);
  }
};

template <typename T>
struct AllocTrait<T *> {

  // Allocate raw buffer to hold the element.
  GEANT_HOST_DEVICE
  static T **Allocate(size_t nElems) { return reinterpret_cast<T **>(new char[nElems * sizeof(T *)]); }

  // Release raw buffer to hold the element.
  GEANT_HOST_DEVICE
  static void Deallocate(T **startBuffer) { delete[]((char *)startBuffer); }

  GEANT_HOST_DEVICE
  static void Destroy(T *&) {}

  GEANT_HOST_DEVICE
  static void Destroy(T ** /*arr*/, size_t /*nElem*/) {}
};
} // namespace Internal

template <typename Type>
class ProxyVector {

protected:
  Type *fData;
  size_t fSize, fMemorySize;
  bool fAllocated;

public:
  using value_type = Type;

  GEANT_HOST_DEVICE
  ProxyVector() : ProxyVector(5) {}

  GEANT_HOST_DEVICE
  ProxyVector(size_t maxsize) : fData(nullptr), fSize(0), fMemorySize(0), fAllocated(true) { reserve(maxsize); }

  GEANT_HOST_DEVICE
  ProxyVector(Type *const vec, const int sz) : fData(vec), fSize(sz), fMemorySize(sz), fAllocated(false) {}

  GEANT_HOST_DEVICE
  ProxyVector(Type *const vec, const int sz, const int maxsize)
      : fData(vec), fSize(sz), fMemorySize(maxsize), fAllocated(false)
  {
  }

  GEANT_HOST_DEVICE
  ProxyVector(ProxyVector const &other) : fSize(other.fSize), fMemorySize(other.fMemorySize), fAllocated(true)
  {
    fData = Internal::AllocTrait<Type>::Allocate(fMemorySize);
    for (size_t i = 0; i < fSize; ++i)
      new (&fData[i]) Type(other.fData[i]);
  }

  GEANT_HOST_DEVICE
  ProxyVector &operator=(ProxyVector const &other)
  {
    if (&other != this) {
      reserve(other.fMemorySize);
      for (size_t i = 0; i < other.fSize; ++i)
        push_back(other.fData[i]);
    }
    return *this;
  }

  GEANT_HOST_DEVICE
  ProxyVector(std::initializer_list<Type> entries)
  {
    fSize       = entries.size();
    fData       = Internal::AllocTrait<Type>::Allocate(fSize);
    fAllocated  = true;
    fMemorySize = entries.size() * sizeof(Type);
    for (auto itm : entries)
      this->push_back(itm);
  }

  GEANT_HOST_DEVICE
  ~ProxyVector()
  {
    if (fAllocated) Internal::AllocTrait<Type>::Deallocate(fData);
  }

  GEANT_HOST_DEVICE
  void clear()
  {
    Internal::AllocTrait<Type>::Destroy(fData, fSize);
    fSize = 0;
  }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  Type Get(const int index) { return fData[index]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  Type &operator[](const int index) { return fData[index]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  Type const &operator[](const int index) const { return fData[index]; }

  GEANT_HOST_DEVICE
  void push_back(const Type item)
  {
    if (fSize == fMemorySize) {
      assert(fAllocated && "Trying to push on a 'fixed' size vector (memory "
                           "not allocated by Vector itself)");
      reserve(fMemorySize << 1);
    }
    new (&fData[fSize]) Type(item);
    fSize++;
  }

  typedef Type *iterator;
  typedef Type const *const_iterator;

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  iterator begin() const { return &fData[0]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  iterator end() const { return &fData[fSize]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  const_iterator cbegin() const { return &fData[0]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  const_iterator cend() const { return &fData[fSize]; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  size_t size() const { return fSize; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  size_t capacity() const { return fMemorySize; }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  void resize(size_t newsize, Type value)
  {
    if (newsize <= fSize) {
      for (size_t i = newsize; i < fSize; ++i) {
        Internal::AllocTrait<Type>::Destroy(fData[i]);
      }
      fSize = newsize;
    } else {
      if (newsize > fMemorySize) {
        reserve(newsize);
      }
      for (size_t i = fSize; i < newsize; ++i)
        push_back(value);
    }
  }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  void reserve(size_t newsize)
  {
    if (newsize <= fMemorySize) {
      // Do nothing ...
    } else {
      Type *newdata = Internal::AllocTrait<Type>::Allocate(newsize);
      for (size_t i = 0; i < fSize; ++i)
        new (&newdata[i]) Type(fData[i]);
      Internal::AllocTrait<Type>::Destroy(fData, fSize);
      if (fAllocated) {
        Internal::AllocTrait<Type>::Deallocate(fData);
      }
      fData       = newdata;
      fMemorySize = newsize;
      fAllocated  = true;
    }
  }

  GEANT_HOST_DEVICE
  GEANT_FORCE_INLINE
  iterator erase(const_iterator position)
  {
    iterator where = (begin() + (position - cbegin()));
    if (where + 1 != end()) {
      auto last = cend();
      for (auto c = where; (c + 1) != last; ++c)
        *c = *(c + 1);
    }
    --fSize;
    if (fSize) Internal::AllocTrait<Type>::Destroy(fData[fSize]);
    return where;
  }
};

} // namespace geantx

