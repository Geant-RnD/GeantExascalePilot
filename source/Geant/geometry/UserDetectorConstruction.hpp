#pragma once
//===--- UserDetectorConstruction.h - Geant-V --------------------------*- C++ -*-===//
//
//                     Geant-V Prototype
//
//===----------------------------------------------------------------------===//
/**
 * @file UserDetectorConstruction.h
 * @brief Implementation of user detector construction in Geant-V prototype
 * @author Andrei Gheata
 */
//===----------------------------------------------------------------------===//

#include "Geant/core/Typedefs.hpp"

#ifdef USE_ROOT
class TGeoMaterial;
#endif

namespace geant {
inline namespace GEANT_IMPL_NAMESPACE {

class RunManager;
class TaskBroker;
class UserFieldConstruction;

/** @brief UserDetectorConstruction class */
class UserDetectorConstruction {
private:
  RunManager *fRunMgr                       = nullptr;
  UserFieldConstruction *fFieldConstruction = nullptr;
// Material conversion callback function
#ifdef USE_ROOT
  static std::function<void *(TGeoMaterial const *)> CreateMaterialConversion();
#endif

protected:
  /** @brief Import regions if available in ROOT geometry*/
  static int ImportRegions();

public:
  /** @brief UserDetectorConstruction constructor */
  UserDetectorConstruction(RunManager *runmgr) { fRunMgr = runmgr; }

  /** @brief UserDetectorConstruction destructor */
  virtual ~UserDetectorConstruction() {}

  /** @brief Creation of materials (optional) */
  virtual void CreateMaterials() {}

  /** @brief Creation of geometry (mandatory) */
  virtual void CreateGeometry() = 0;

  /** Setup geometry connections to user indices */
  int SetupGeometry(vector_t<Volume_t const *> &volumes, TaskBroker *broker = nullptr);

  /** Load geometry from a file. Can be called from the user-defined CreateGeometry */
  bool LoadGeometry(const char *filename);

  /** Object which will create the EM field and the solvers for tracking in it */
  void SetUserFieldConstruction(UserFieldConstruction *ufc)
  {
    fFieldConstruction = ufc;
    // fInitialisedRKIntegration= false;  //  Needs to be re-done !!
  }

  /** obtain the field c-tion object */
  UserFieldConstruction *GetFieldConstruction() { return fFieldConstruction; }
  const UserFieldConstruction *GetFieldConstruction() const { return fFieldConstruction; }

  // these methods will become private and non-static after migration to
  // user detector construction is complete
  static bool LoadVecGeomGeometry(TaskBroker *broker = nullptr);
  static void InitNavigators();
};

} // GEANT_IMPL_NAMESPACE
} // Geant
