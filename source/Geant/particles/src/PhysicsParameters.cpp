
#include "Geant/particles/PhysicsParameters.hpp"

#include "Geant/core/SystemOfUnits.hpp"

#include "Geant/core/Logger.hpp"

#include "Geant/core/math_wrappers.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace geantx {
std::vector<PhysicsParameters *> PhysicsParameters::gThePhysicsParametersTable;

// set to default values
double PhysicsParameters::gMinAllowedGammaCutEnergy    = 990.0 * geantx::units::eV;
double PhysicsParameters::gMaxAllowedGammaCutEnergy    = 10.0 * geantx::units::GeV;
double PhysicsParameters::gMinAllowedElectronCutEnergy = 990.0 * geantx::units::eV;
double PhysicsParameters::gMaxAllowedElectronCutEnergy = 10.0 * geantx::units::GeV;
double PhysicsParameters::gMinAllowedPositronCutEnergy = 990.0 * geantx::units::eV;
double PhysicsParameters::gMaxAllowedPositronCutEnergy = 10.0 * geantx::units::GeV;

double PhysicsParameters::gDefaultGammaCutInLength    = 1.0 * geantx::units::mm;
double PhysicsParameters::gDefaultElectronCutInLength = 1.0 * geantx::units::mm;
double PhysicsParameters::gDefaultPositronCutInLength = 1.0 * geantx::units::mm;

double PhysicsParameters::gDefaultGammaCutInEnergy    = 100.0 * geantx::units::keV;
double PhysicsParameters::gDefaultElectronCutInEnergy = 100.0 * geantx::units::keV;
double PhysicsParameters::gDefaultPositronCutInEnergy = 100.0 * geantx::units::keV;

PhysicsParameters::PhysicsParameters()
{
  fMinLossTableEnergy        = 100.0 * geantx::units::eV;
  fMaxLossTableEnergy        = 100.0 * geantx::units::TeV;
  fNumLossTableBins          = 84;
  fNumLossTableBinsPerDecade = 7;

  fIsComputeCSDARange = false;

  fMinLambdaTableEnergy        = fMinLossTableEnergy;
  fMaxLambdaTableEnergy        = fMaxLossTableEnergy;
  fNumLambdaTableBins          = fNumLossTableBins;
  fNumLambdaTableBinsPerDecade = fNumLossTableBinsPerDecade;

  fLowestElectronTrackingEnergy = 1.0 * geantx::units::keV;

  fLinearEnergyLossLimit = 0.01;

  fDRoverRange = 0.2;
  fFinalRange  = 1.0 * geantx::units::mm;

  gThePhysicsParametersTable.push_back(this);
}

const PhysicsParameters *PhysicsParameters::GetPhysicsParametersForRegion(int regionindx)
{
  PhysicsParameters *physPar = nullptr;
  for (auto &itr : gThePhysicsParametersTable) {
    if (itr->IsActiveRegion(regionindx)) {
      physPar = itr;
      break;
    }
  }
  return physPar;
}

void PhysicsParameters::Clear()
{
  for (auto &itr : gThePhysicsParametersTable) {
    delete itr;
  }
  gThePhysicsParametersTable.clear();
}

void PhysicsParameters::SetMinLossTableEnergy(double val)
{
  if (val > 1.e-3 * geantx::units::eV && val < fMaxLossTableEnergy) {
    fMinLossTableEnergy = val;
    // update number of bins
    fNumLossTableBins =
        fNumLossTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLossTableEnergy / fMinLossTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetMinLossTableEnergy() " << std::endl
        << "    Value of fMinLossTableEnergy is out of range: "
        << val / geantx::units::keV << " [keV] so it's ignored!" << std::endl;
  }
}

void PhysicsParameters::SetMaxLossTableEnergy(double val)
{
  if (val > fMinLossTableEnergy && val < 1.e+7 * geantx::units::TeV) {
    fMaxLossTableEnergy = val;
    // update number of bins
    fNumLossTableBins =
        fNumLossTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLossTableEnergy / fMinLossTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetMaxLossTableEnergy() " << std::endl
        << "    Value of fMaxLossTableEnergy is out of range: "
        << val / geantx::units::GeV << " [GeV] so it's ignored!" << std::endl;
  }
}

// minimum 5 for the better constrained splie
void PhysicsParameters::SetNumLossTableBins(int val)
{
  if (val > 4 && val < 10000000) {
    fNumLossTableBins          = val;
    fNumLossTableBinsPerDecade = std::lrint(
        fNumLossTableBins / Math::Log10(fMaxLossTableEnergy / fMinLossTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetNumLossTableBins() " << std::endl
        << "    Value of fNumLossTableBins is out of range: " << val
        << " so it's ignored!" << std::endl;
  }
}

// minimum 5 for the better constrained splie
void PhysicsParameters::SetNumLossTableBinsPerDecade(int val)
{
  if (val > 4 && val < 1000000) {
    fNumLossTableBinsPerDecade = val;
    fNumLossTableBins =
        fNumLossTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLossTableEnergy / fMinLossTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetNumLossTableBinsPerDecade() " << std::endl
        << "    Value of fNumLossTableBinsPerDecade is out of range: " << val
        << " so it's ignored!" << std::endl;
  }
}

////
void PhysicsParameters::SetMinLambdaTableEnergy(double val)
{
  if (val > 1.e-3 * geantx::units::eV && val < fMaxLambdaTableEnergy) {
    fMinLambdaTableEnergy = val;
    // update number of bins
    fNumLambdaTableBins =
        fNumLambdaTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLambdaTableEnergy / fMinLambdaTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetMinLambdaTableEnergy() " << std::endl
        << "    Value of fMinLambdaTableEnergy is out of range: "
        << val / geantx::units::keV << " [keV] so it's ignored!" << std::endl;
  }
}

void PhysicsParameters::SetMaxLambdaTableEnergy(double val)
{
  if (val > fMinLambdaTableEnergy && val < 1.e+7 * geantx::units::TeV) {
    fMaxLambdaTableEnergy = val;
    // update number of bins
    fNumLambdaTableBins =
        fNumLambdaTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLambdaTableEnergy / fMinLambdaTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetMaxLambdaTableEnergy() " << std::endl
        << "    Value of fMaxLambdaTableEnergy is out of range: "
        << val / geantx::units::GeV << " [GeV] so it's ignored!" << std::endl;
  }
}

// minimum 5 for the better constrained splie
void PhysicsParameters::SetNumLambdaTableBins(int val)
{
  if (val > 4 && val < 10000000) {
    fNumLambdaTableBins          = val;
    fNumLambdaTableBinsPerDecade = std::lrint(
        fNumLambdaTableBins / Math::Log10(fMaxLambdaTableEnergy / fMinLambdaTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetNumLambdaTableBins() " << std::endl
        << "    Value of fNumLambdaTableBins is out of range: " << val
        << " so it's ignored!" << std::endl;
  }
}

// minimum 5 for the better constrained splie
void PhysicsParameters::SetNumLambdaTableBinsPerDecade(int val)
{
  if (val > 4 && val < 1000000) {
    fNumLambdaTableBinsPerDecade = val;
    fNumLambdaTableBins =
        fNumLambdaTableBinsPerDecade *
        std::lrint(Math::Log10(fMaxLambdaTableEnergy / fMinLambdaTableEnergy));
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetNumLambdaTableBinsPerDecade() " << std::endl
        << "    Value of fNumLambdaTableBinsPerDecade is out of range: " << val
        << " so it's ignored!" << std::endl;
  }
}

void PhysicsParameters::SetLowestElectronTrackingEnergy(double val)
{
  if (val >= 0.0) {
    fLowestElectronTrackingEnergy = val;
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetLowestElectronTrackingEnergy " << std::endl
        << "    Value of fLowestElectronTrackingEnergy is out of range: "
        << val / geantx::units::keV << " [keV] so it's ignored!" << std::endl;
  }
}

void PhysicsParameters::SetLinearEnergyLossLimit(double val)
{
  if (val > 0.0 && val < 0.5) {
    fLinearEnergyLossLimit = val;
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetLinearEnergyLossLimit " << std::endl
        << "    Value of fLinearEnergyLossLimit is out of range: " << val
        << " so it's ignored!" << std::endl;
  }
}

void PhysicsParameters::SetStepFunction(double roverrange, double finalrange)
{
  if (roverrange > 0.0 && roverrange <= 1.0 && finalrange > 0.0) {
    fDRoverRange = roverrange;
    fFinalRange  = finalrange;
  } else {
    geantx::Log(geantx::kWarning)
        << "PhysicsParameters::SetStepFunction " << std::endl
        << "    Values of step function are out of range: " << roverrange << ", "
        << finalrange / geantx::units::mm << " [mm]  so they are ignored!" << std::endl;
  }
}

std::ostream &operator<<(std::ostream &flux, PhysicsParameters *physpar)
{
  flux << *physpar;
  return flux;
}

std::ostream &operator<<(std::ostream &flux, PhysicsParameters &physpar)
{
  using geantx::units::cm;
  using geantx::units::GeV;
  //  using geantx::units::eV;
  int textW = 70;
  int valW  = 12;

  flux << std::setw(100) << std::setfill('=') << std::left << "  =" << std::endl
       << std::setw(40) << std::setfill('-') << std::left << "  -"
       << " Physics parameters " << std::setw(40) << std::setfill('-') << std::left << "-"
       << std::endl
       << std::setw(100) << std::setfill('=') << std::left << "  =" << std::setfill(' ')
       << std::endl;

  // parameters that are the same in each region
  flux << "    **** (Static)Parameters that must have the same value in each region: \n"
       << "     - fMinAllowedGammaCutEnergy    =   " << std::setw(10)
       << PhysicsParameters::GetMinAllowedGammaCutEnergy() / GeV << " [GeV]\n"
       << "     - fMaxAllowedGammaCutEnergy    =   " << std::setw(10)
       << PhysicsParameters::GetMaxAllowedGammaCutEnergy() / GeV << " [GeV]\n"
       << "     - fMinAllowedElectronCutEnergy =   " << std::setw(10)
       << PhysicsParameters::GetMinAllowedElectronCutEnergy() / GeV << " [GeV]\n"
       << "     - fMaxAllowedElectronCutEnergy =   " << std::setw(10)
       << PhysicsParameters::GetMaxAllowedElectronCutEnergy() / GeV << " [GeV]\n"
       << "     - fMinAllowedPositronCutEnergy =   " << std::setw(10)
       << PhysicsParameters::GetMinAllowedPositronCutEnergy() / GeV << " [GeV]\n"
       << "     - fMaxAllowedPositronCutEnergy =   " << std::setw(10)
       << PhysicsParameters::GetMaxAllowedPositronCutEnergy() / GeV << " [GeV]\n"
       << "     - fDefaultGammaCutInLength     =   " << std::setw(10)
       << PhysicsParameters::GetDefaultGammaCutInLength() / cm << "  [cm]\n"
       << "     - fDefaultElectronCutInLength  =   " << std::setw(10)
       << PhysicsParameters::GetDefaultElectronCutInLength() / cm << "  [cm]\n"
       << "     - fDefaultPositronCutInLength  =   " << std::setw(10)
       << PhysicsParameters::GetDefaultPositronCutInLength() / cm << "  [cm]\n"
       << "     - fDefaultGammaCutInEnergy     =   " << std::setw(10)
       << PhysicsParameters::GetDefaultGammaCutInEnergy() / GeV << " [GeV]\n"
       << "     - fDefaultElectronCutInEnergy  =   " << std::setw(10)
       << PhysicsParameters::GetDefaultElectronCutInEnergy() / GeV << " [GeV]\n"
       << "     - fDefaultPositronCutInEnergy  =   " << std::setw(10)
       << PhysicsParameters::GetDefaultPositronCutInEnergy() / GeV << " [GeV]\n\n";

  flux << "    **** Active in regions: ";
  for (unsigned i = 0; i < physpar.GetListActiveRegions().size(); ++i) {
    if (physpar.IsActiveRegion(i)) flux << std::setw(10) << i;
  }
  flux << "\n\n";

  flux << "    **** Additional parameter values: \n\n";
  flux << std::setw(90) << std::left << std::setfill('-')
       << "     --- Energy loss table related parameter values " << std::setfill(' ')
       << "\n";
  flux << std::left << std::setw(textW)
       << "     - Minimum kinetic energy of the energy loss table grid " << std::right
       << std::setw(valW) << physpar.GetMinLossTableEnergy() / geantx::units::keV
       << "   [keV] \n"
       << std::left << std::setw(textW)
       << "     - Maximum kinetic energy of the energy loss table grid " << std::right
       << std::setw(valW) << physpar.GetMaxLossTableEnergy() / geantx::units::TeV
       << "   [TeV] \n"
       << std::left << std::setw(textW)
       << "     - Number of energy bins in the energy loss table " << std::right
       << std::setw(valW) << physpar.GetNumLossTableBins() << "\n"
       << std::left << std::setw(textW)
       << "     - Number of energy bins per decade in the energy loss table "
       << std::right << std::setw(valW) << physpar.GetNumLossTableBinsPerDecade() << "\n"
       << std::endl;

  flux << std::setw(90) << std::left << std::setfill('-')
       << "     --- Lambda table related parameter values " << std::setfill(' ') << "\n";
  flux << std::left << std::setw(textW)
       << "     - Minimum kinetic energy of the lambda table grid " << std::right
       << std::setw(valW) << physpar.GetMinLambdaTableEnergy() / geantx::units::keV
       << "   [keV] \n"
       << std::left << std::setw(textW)
       << "     - Maximum kinetic energy of the lambda table grid " << std::right
       << std::setw(valW) << physpar.GetMaxLambdaTableEnergy() / geantx::units::TeV
       << "   [TeV] \n"
       << std::left << std::setw(textW)
       << "     - Number of energy bins in the lambda table " << std::right
       << std::setw(valW) << physpar.GetNumLambdaTableBins() << "\n"
       << std::left << std::setw(textW)
       << "     - Number of energy bins per decade in the lambda table " << std::right
       << std::setw(valW) << physpar.GetNumLambdaTableBinsPerDecade() << "\n"
       << std::endl;

  flux << std::setw(90) << std::left << std::setfill('-')
       << "     --- Electron/positron related parameters " << std::setfill(' ') << "\n";
  flux << std::left << std::setw(textW)
       << "     - Lowest e-/e+ kinetic energy for tracking  " << std::right
       << std::setw(valW)
       << physpar.GetLowestElectronTrackingEnergy() / geantx::units::keV << "   [keV] \n"
       << std::endl;

  flux << std::setw(90) << std::left << std::setfill('-') << "     --- Other parameters "
       << std::setfill(' ') << "\n";
  flux << std::left << std::setw(textW) << "     - Compute CSDA range tables  "
       << std::right << std::setw(valW) << physpar.GetIsComputeCSDARange() << "\n"
       << std::left << std::setw(textW)
       << "     - Linear energy loss approximation limit  " << std::right
       << std::setw(valW) << physpar.GetLinearEnergyLossLimit() << "\n"
       << std::left << std::setw(textW)
       << "     - Continuous step limit function parameters : fDRoverRange " << std::right
       << std::setw(valW) << physpar.GetDRoverRange() << "\n"
       << std::left << std::setw(textW)
       << "     - Continuous step limit function parameters : fFInalRange " << std::right
       << std::setw(valW) << physpar.GetFinalRange() / geantx::units::mm << "   [mm] "
       << std::endl;

  flux << std::setw(100) << std::setfill('=') << std::left << "  =" << std::setfill(' ')
       << std::endl
       << std::endl;

  return flux;
}

} // namespace geantx
