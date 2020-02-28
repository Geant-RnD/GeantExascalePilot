#include "Geant/geometry/UserDetectorConstruction.hpp"

#include "Geant/core/Logger.hpp"
//#include "Geant/core/Region.h"
//#include "Geant/TaskBroker.h"
//#include "Geant/VBconnector.h"
//#include "Geant/RunManager.h"

#include "Geant/material/Element.hpp"
#include "Geant/material/Material.hpp"
#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxLevelLocator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/VNavigator.h"
#ifdef USE_ROOT
#  include "TGeoRegion.h"
#  include "VecGeom/management/RootGeoManager.h"
#endif
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

//______________________________________________________________________________
int UserDetectorConstruction::SetupGeometry(vector_t<Volume_t const *> &volumes,
                                            TaskBroker *broker)
{
  // Setup geometry after being constructed by the user (or loaded from file)
  int nvolumes = 0;
  LoadVecGeomGeometry(broker);
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(volumes);
  nvolumes = volumes.size();
  if (!nvolumes) {
    geantx::Log(kFatal).From("UserDetectorConstruction::SetupGeometry")
        << "Geometry is empty";
    return 0;
  }

  // for (auto i = 0; i < nvolumes; ++i) {
  //   Volume_t *vol          = (Volume_t *)volumes[i];
  //   VBconnector *connector = new VBconnector(i);
  //   vol->SetBasketManagerPtr(connector);
  // }
  return nvolumes;
}

//______________________________________________________________________________
bool UserDetectorConstruction::LoadGeometry(const char *filename)
{
// Load geometry from file. This is callable from the user detector construction class.
#ifdef USE_ROOT
  if (gGeoManager || TGeoManager::Import(filename)) return true;
#else
  vecgeom::GeoManager *geom = &vecgeom::GeoManager::Instance();
  if (geom) {
    geom->LoadGeometryFromSharedLib(filename);
    return true;
  }
#endif
  geantx::Log(kError).From("UserDetectorConstruction::LoadGeometry")
      << "Cannot load geometry from file " << filename;
  return false;
}

//______________________________________________________________________________
bool UserDetectorConstruction::LoadVecGeomGeometry(TaskBroker *broker)
{
  if (vecgeom::GeoManager::Instance().GetWorld() == NULL) {
#ifdef USE_ROOT
    vecgeom::RootGeoManager::Instance().SetMaterialConversionHook(
        CreateMaterialConversion());
    geantx::Log(kStatus) << "Now loading VecGeom geometry...";
    vecgeom::RootGeoManager::Instance().LoadRootGeometry();
    geantx::LogLocal(kStatus) << "Loading VecGeom geometry done";
    std::vector<vecgeom::LogicalVolume *> v1;
    vecgeom::GeoManager::Instance().GetAllLogicalVolumes(v1);
    geantx::LogLocal(kStatus) << "--- Have logical volumes " << v1.size();
    std::vector<vecgeom::VPlacedVolume *> v2;
    vecgeom::GeoManager::Instance().getAllPlacedVolumes(v2);
    geantx::LogLocal(kStatus) << "--- Have placed volumes " << v2.size();
    //    vecgeom::RootGeoManager::Instance().world()->PrintContent();
    // Create regions if any available in the ROOT geometry
    int nregions = UserDetectorConstruction::ImportRegions();
    geantx::LogLocal(kStatus) << "--- Imported %d regions" << nregions;
#endif
  }
  if (broker) {
    geantx::Log(kStatus) << "Now uploading VecGeom geometry to Coprocessor(s)...";
    // return broker->UploadGeometry();
  }
  InitNavigators();
  return true;
}

//______________________________________________________________________________
int UserDetectorConstruction::ImportRegions()
{
  // Import regions if available in TGeo
  int nregions = 0;
#if defined(USE_ROOT)
  using Region        = vecgeom::Region;
  using LogicalVolume = vecgeom::LogicalVolume;

  double electronCut, positronCut, gammaCut, protonCut;
  // Loop on ROOT regions if any
  nregions = gGeoManager->GetNregions();
  for (int i = 0; i < nregions; ++i) {
    TGeoRegion *region_root = gGeoManager->GetRegion(i);
    electronCut = positronCut = gammaCut = protonCut = -1.;
    // loop cuts for this region
    for (int icut = 0; icut < region_root->GetNcuts(); ++icut) {
      std::string cutname = region_root->GetCut(icut)->GetName();
      double cutvalue     = region_root->GetCut(icut)->GetCut();
      if (cutname == "gamcut") gammaCut = cutvalue;
      if (cutname == "ecut") electronCut = cutvalue;
      if (cutname == "poscut") positronCut = cutvalue;
      if (cutname == "pcut") protonCut = cutvalue;
    }
    Region *region = new Region(std::string(region_root->GetName()), gammaCut,
                                electronCut, positronCut, protonCut);

    geantx::LogLocal(kStatus) << "Created region " << region_root->GetName()
                              << " with: gammaCut = " << gammaCut
                              << " [cm], eleCut = " << electronCut
                              << " [cm], posCut = " << positronCut
                              << " [cm], protonCut = " << protonCut << " [cm]";

    // loop volumes in the region. Volumes should be already converted to LogicalVolume
    for (int ivol = 0; ivol < region_root->GetNvolumes(); ++ivol) {
      LogicalVolume *vol =
          vecgeom::RootGeoManager::Instance().Convert(region_root->GetVolume(ivol));
      vol->SetRegion(region);
      // geantx::LogLocal(kStatus) << "   added to volume %s\n", vol->GetName());
    }
  }
#endif
  return nregions;
}

//______________________________________________________________________________
void UserDetectorConstruction::InitNavigators()
{
  for (auto &lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap()) {
    if (lvol.second->GetDaughtersp()->size() < 4) {
      lvol.second->SetNavigator(vecgeom::NewSimpleNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 5) {
      lvol.second->SetNavigator(vecgeom::SimpleABBoxNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 10) {
      lvol.second->SetNavigator(vecgeom::HybridNavigator<>::Instance());
      vecgeom::HybridManager2::Instance().InitStructure((lvol.second));
    }
    lvol.second->SetLevelLocator(vecgeom::SimpleABBoxLevelLocator::GetInstance());
  }
}

//______________________________________________________________________________
#ifdef USE_ROOT
std::function<void *(TGeoMaterial const *)> UserDetectorConstruction::
    CreateMaterialConversion()
{
  return [](TGeoMaterial const *rootmat) {
    // geantx::Log(geantx::kStatus) <<"     -->  Creating Material  "<<rootmat->GetName();
    int numElem = rootmat->GetNelements();
    double density =
        rootmat->GetDensity() * geant::units::g / geant::units::cm3; // in g/cm3
    const std::string name = rootmat->GetName();
    // check if it is a G4 NIST material
    std::string postName = "";
    bool isNistMaterial  = false;
    if (name.substr(0, 3) == "G4_") {
      postName       = name.substr(3);
      isNistMaterial = true;
    }
    geantx::Material *gmat = nullptr;
    if (isNistMaterial) {
      std::string nistName = "NIST_MAT_" + postName;
      gmat                 = geantx::Material::NISTMaterial(nistName);
    } else {
      // find or create material
      gmat = geantx::Material::GetMaterial(name);
      if (gmat) {
        // geantx::Log(geantx::kStatus) << " Material "<<name << " has already been
        // created.!"<< std::endl;
        return gmat;
      }
      gmat = new geantx::Material(name, density, numElem);
      for (int j = 0; j < numElem; ++j) {
        double va;
        double vz;
        double vw;
        const_cast<TGeoMaterial *>(rootmat)->GetElementProp(va, vz, vw, j);
        // create NIST element
        geantx::Element *elX = geantx::Element::NISTElement(vz);
        // add to the Material
        gmat->AddElement(elX, vw);
      }
    }
    gmat->SetIsUsed(true);
    return gmat;
  };
}
#endif

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
