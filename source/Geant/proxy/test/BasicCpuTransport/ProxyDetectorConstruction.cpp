
#include "ProxyDetectorConstruction.hpp"

#include "Geant/geometry/RunManager.hpp"
#include "Geant/geometry/UserDetectorConstruction.hpp"

#include <iostream>
#include <vector>

// Material includes
#include "Geant/material/Element.hpp"
#include "Geant/material/Material.hpp"
#include "Geant/material/MaterialProperties.hpp"
//#include "Geant/NISTElementData.hpp"
#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/core/SystemOfUnits.hpp"

// Region and productionCut includes
//#include "Geant/Region.hpp"
//#include "Geant/PhysicsParameters.h"
//#include "Geant/MaterialCuts.h"
/////////////////////////////////
// VecGeom includes
#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/LogicalVolume.h"
/////////////////////////////////

using namespace geantx::units;
// using mm = geantx::units::m;
// using g = geantx::units::g;
// using mole = geantx::units::mole;
// using pascal = geantx::units::pascal;
// using kelvin = geantx::units::kelvin;

namespace userapplication {
ProxyDetectorConstruction::ProxyDetectorConstruction(geantx::RunManager *runmgr)
    : geantx::UserDetectorConstruction(runmgr)
{
  fNumberOfAbsorbers      = 2;
  fAbsorberThicknesses[0] = 2.3 * mm;
  fAbsorberThicknesses[1] = 5.7 * mm;
  fNumberOfLayers         = 50;
  fCaloSizeYZ             = 40. * cm;
  //
  ComputeCalorimeter();
  //
  fWorldMaterialName        = "Galactic";
  fAbsorberMaterialNames[0] = "NIST_MAT_Pb";
  fAbsorberMaterialNames[1] = "NIST_MAT_lAr";
  //
  fProductionCut = 0.7 * mm;
  //
  for (int i = 0; i < gMaxNumAbsorbers; ++i) {
    fAbsorberLogicVolIDs[i] = -1;
    fAbsorberMaterials[i]   = nullptr;
  }
  fWorldMaterial = nullptr;
}

ProxyDetectorConstruction::~ProxyDetectorConstruction()
{ /* nothing to do */
}

void ProxyDetectorConstruction::SetNumberOfAbsorbersPerLayer(int numabs)
{
  if (numabs > gMaxNumAbsorbers) {
    std::cerr << "  **** ERROR: "
                 "ProxyDetectorConstruction::SetNumberOfAbsorbersPerLayer() \n"
                 "    Number of absorbers per layer "
              << numabs << " > maximum " << gMaxNumAbsorbers << std::endl;
    exit(-1);
  }
  fNumberOfAbsorbers = numabs;
}

void ProxyDetectorConstruction::SetAbsorberThickness(int absindx, double thick)
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr << "  **** ERROR: ProxyDetectorConstruction::SetAbsorberThickness() \n"
                 "    Unknown absorber index = "
              << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )"
              << std::endl;
    exit(-1);
  }
  fAbsorberThicknesses[absindx] = thick;
}

double ProxyDetectorConstruction::GetAbsorberThickness(int absindx) const
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr << "  **** ERROR: ProxyDetectorConstruction::GetAbsorberThickness() \n"
                 "    Unknown absorber index = "
              << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )"
              << std::endl;
    exit(-1);
  }
  return fAbsorberThicknesses[absindx];
}

void ProxyDetectorConstruction::SetAbsorberMaterialName(int absindx,
                                                        const std::string &matname)
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr << "  **** ERROR: ProxyDetectorConstruction::SetAbsorberMaterialName() \n"
                 "    Unknown absorber index = "
              << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )"
              << std::endl;
    exit(-1);
  }
  fAbsorberMaterialNames[absindx] = matname;
}

void ProxyDetectorConstruction::SetAbsorberMaterial(int absindx,
                                                    const std::string &matname)
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr << "  **** ERROR: ProxyDetectorConstruction::SetAbsorberMaterial() \n"
                 "    Unknown absorber index = "
              << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )"
              << std::endl;
    exit(-1);
  }
  // will report proper error msg if the material cannot be found or created
  geantx::Material *mat       = geantx::Material::NISTMaterial(matname);
  fAbsorberMaterials[absindx] = mat;
}

const geantx::Material *ProxyDetectorConstruction::GetAbsorberMaterial(int absindx) const
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr << "  **** ERROR: ProxyDetectorConstruction::GetAbsorberMaterial() \n"
                 "    Unknown absorber index = "
              << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )"
              << std::endl;
    exit(-1);
  }
  return fAbsorberMaterials[absindx];
}

int ProxyDetectorConstruction::GetAbsorberLogicalVolumeID(int absindx) const
{
  if (absindx >= fNumberOfAbsorbers) {
    std::cerr
        << "  **** ERROR: ProxyDetectorConstruction::GetAbsorberLogicalVolumeID() \n"
           "    Unknown absorber index = "
        << absindx << " ( should be <= " << fNumberOfAbsorbers - 1 << " )" << std::endl;
    exit(-1);
  }
  return fAbsorberLogicVolIDs[absindx];
}

void ProxyDetectorConstruction::ComputeCalorimeter()
{
  fLayerThickness = 0.0;
  for (int i = 0; i < fNumberOfAbsorbers; ++i) {
    fLayerThickness += fAbsorberThicknesses[i];
  }
  fCaloSizeX   = fNumberOfLayers * fLayerThickness;
  fWorldSizeYZ = 1.2 * fCaloSizeYZ;
  fWorldSizeX  = 1.2 * fCaloSizeX;
}

void ProxyDetectorConstruction::DetectorInfo()
{
  std::cout << "\n ========================    Detector Info   "
               "========================================  "
            << std::endl;
  std::cout << "     Calorimeter is " << fNumberOfLayers << " layers of [ ";
  for (int i = 0; i < fNumberOfAbsorbers; ++i) {
    std::cerr << fAbsorberThicknesses[i] / mm << "  mm " << fAbsorberMaterialNames[i];
    if (i < fNumberOfAbsorbers - 1) std::cerr << " + ";
  }
  std::cout << " ]" << std::endl;
  std::cout << " ========================================================================"
               "============  \n"
            << std::endl;
}

void ProxyDetectorConstruction::CreateMaterials()
{
  geantx::Element *elH  = geantx::Element::NISTElement(1);
  geantx::Element *elC  = geantx::Element::NISTElement(6);
  geantx::Element *elPb = geantx::Element::NISTElement(82);
  //
  // create material Galactic
  double density     = kUniverseMeanDensity;
  double pressure    = 3.e-18 * pascal;
  double temperature = 2.73 * kelvin;
  double z           = 1.;
  double a           = 1.008 * g / mole;
  new geantx::Material("Galactic", z, a, density, geantx::MaterialState::kStateGas,
                       temperature, pressure);
  //
  // create material Lead (by mass fraction)
  double massfraction       = 1.;
  int ncomponents           = 1;
  density                   = 11.35 * g / cm3;
  geantx::Material *matLead = new geantx::Material("Lead", density, ncomponents);
  matLead->AddElement(elPb, massfraction = 1.0);
  //
  // create Scintillator (by atom count)
  density                  = 1.032 * g / cm3;
  ncomponents              = 2;
  int natoms               = 1;
  geantx::Material *matSci = new geantx::Material("Scintillator", density, ncomponents);
  matSci->AddElement(elC, natoms = 9);
  matSci->AddElement(elH, natoms = 10);
  //
  // create liquidArgon (from material NIST_MAT_lAr with different density by mass
  // fraction)
  geantx::Material *matlAr = geantx::Material::NISTMaterial("NIST_MAT_lAr");
  density                  = 1.390 * g / cm3;
  ncomponents              = 1;
  massfraction             = 1.;
  geantx::Material *lArEm3 = new geantx::Material("liquidArgon", density, ncomponents);
  lArEm3->AddMaterial(matlAr, massfraction);
}

void ProxyDetectorConstruction::CreateGeometry()
{
  // compute possible updated calorimeter properties
  ComputeCalorimeter();
  // first set the materials that have been defined by their name
  fWorldMaterial = geantx::Material::NISTMaterial(fWorldMaterialName);
  for (int i = 0; i < fNumberOfAbsorbers; ++i) {
    SetAbsorberMaterial(i, fAbsorberMaterialNames[i]);
  }

  // // create one region (with the production cut in length)
  // // all logical volume will be assigned to this region (i.e. enough to set for the
  // world) vecgeom::Region *aRegion = new vecgeom::Region("Region", true,
  // fProductionCut, fProductionCut, fProductionCut);

  // create geometry
  vecgeom::UnplacedBox *world =
      new vecgeom::UnplacedBox(0.5 * fWorldSizeX, 0.5 * fWorldSizeYZ, 0.5 * fWorldSizeYZ);
  // create the corresponding logical volume
  vecgeom::LogicalVolume *logicWorld = new vecgeom::LogicalVolume("world", world);
  // set material pointer
  logicWorld->SetMaterialPtr(fWorldMaterial);
  //
  // create layer
  vecgeom::UnplacedBox *layer = new vecgeom::UnplacedBox(
      "layer", 0.5 * fLayerThickness, 0.5 * fCaloSizeYZ, 0.5 * fCaloSizeYZ);
  // create the corresponding logical volume
  vecgeom::LogicalVolume *logicLayer = new vecgeom::LogicalVolume("layer", layer);
  // set material pointer
  logicLayer->SetMaterialPtr(fWorldMaterial);
  //
  // create all absorbers and place them into the layer
  char name[512];
  double xcenter = 0.;
  double xstart  = -0.5 * fLayerThickness;
  for (int i = 0; i < fNumberOfAbsorbers; ++i) {
    // create absorber i
    sprintf(name, "abs_%d", i);
    double sizeX                = 0.5 * fAbsorberThicknesses[i];
    double sizeYZ               = 0.5 * fCaloSizeYZ;
    vecgeom::UnplacedBox *absor = new vecgeom::UnplacedBox(name, sizeX, sizeYZ, sizeYZ);
    // create the corresponding logical volume
    vecgeom::LogicalVolume *logicAbsor = new vecgeom::LogicalVolume(name, absor);
    // set material
    logicAbsor->SetMaterialPtr(fAbsorberMaterials[i]);
    // get the logical volume ID
    fAbsorberLogicVolIDs[i] = logicAbsor->id();
    // compute placement and place the absorber into the layer
    xcenter = xstart + sizeX;
    xstart += fAbsorberThicknesses[i];
    vecgeom::Transformation3D *place =
        new vecgeom::Transformation3D(xcenter, 0, 0, 0, 0, 0);
    logicLayer->PlaceDaughter(name, logicAbsor, place);
  }
  //
  // create calorimeter and place fNumberOfLayers into the calorimeter
  vecgeom::UnplacedBox *calo = new vecgeom::UnplacedBox(
      "calo", 0.5 * fCaloSizeX, 0.5 * fCaloSizeYZ, 0.5 * fCaloSizeYZ);
  // create the corresponding logical volume
  vecgeom::LogicalVolume *logicCalo = new vecgeom::LogicalVolume("calo", calo);
  // set material pointer
  logicCalo->SetMaterialPtr(fWorldMaterial);
  xcenter = 0.;
  xstart  = -0.5 * fCaloSizeX;
  std::vector<int> layerIDs;
  int maxLayerID = -1;
  // place fNumberOfLayers layers into the calorimeter and store the ID-s of the placed
  // layers (as VPlacedVolume-s)
  for (int i = 0; i < fNumberOfLayers; ++i) {
    sprintf(name, "layer_%d", i);
    xcenter = xstart + 0.5 * fLayerThickness;
    xstart += fLayerThickness;
    vecgeom::Transformation3D *place =
        new vecgeom::Transformation3D(xcenter, 0, 0, 0, 0, 0);
    const vecgeom::VPlacedVolume *placedLayer =
        logicCalo->PlaceDaughter(name, logicLayer, place);
    int layerID = placedLayer->id();
    layerIDs.push_back(layerID);
    if (layerID > maxLayerID) {
      maxLayerID = layerID;
    }
  }
  // create the layer (as VPlacedVolume) IDs to their index map
  fLayerIDToLayerIndex.resize(maxLayerID + 1, -1);
  for (int l = 0; l < fNumberOfLayers; ++l) {
    fLayerIDToLayerIndex[layerIDs[l]] = l;
  }
  layerIDs.clear();
  //
  // place the calorimeter into the world
  vecgeom::Transformation3D *place = new vecgeom::Transformation3D(0, 0, 0, 0, 0, 0);
  logicWorld->PlaceDaughter("calorimeter", logicCalo, place);
  //
  // set the world logical volume, assigned it to the created region and close the
  // geometry
  vecgeom::VPlacedVolume *w = logicWorld->Place();
  vecgeom::GeoManager::Instance().SetWorld(w);
  // // set region for the world logical volume (will be set for all daughter volumes)
  // logicWorld->SetRegion(aRegion);
  vecgeom::GeoManager::Instance().CloseGeometry();
}

} // namespace userapplication
