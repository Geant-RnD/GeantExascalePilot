//===------------------ GeantX --------------------------------------------===//
//                                                                              
// Geant Exascale Pilot                                                         
//                                                                              
// For the licensing terms see LICENSE file.                                    
// For the list of contributors see CREDITS file.                               
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.         
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyPhysicalConstants.hpp
 * @brief Physical Constants based on CLHEP PhysicalConstants.h
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/proxy/ProxySystemOfUnits.hpp"

namespace geantx {
namespace clhep {

//
constexpr double Avogadro = 6.02214179e+23/mole;

//
// c   = 299.792458 mm/ns
// c^2 = 898.7404 (mm/ns)^2
//
constexpr double c_light   = 2.99792458e+8 * m/s;
constexpr double c_squared = c_light * c_light;

//
// h     = 4.13566e-12 MeV*ns
// hbar  = 6.58212e-13 MeV*ns
// hbarc = 197.32705e-12 MeV*mm
//
constexpr double h_Planck      = 6.62606896e-34 * joule*s;
constexpr double hbar_Planck   = h_Planck/twopi;
constexpr double hbarc         = hbar_Planck * c_light;
constexpr double hbarc_squared = hbarc * hbarc;

//
constexpr double electron_charge = - eplus; // see SystemOfUnits.h
constexpr double e_squared = eplus * eplus;

//
// amu_c2 - atomic equivalent mass unit
//        - AKA, unified atomic mass unit (u)
// amu    - atomic mass unit
//
constexpr double electron_mass_c2 = 0.510998910 * MeV;
constexpr double inv_electron_mass_c2 = 1.0/electron_mass_c2;
constexpr double   proton_mass_c2 = 938.272013 * MeV;
constexpr double  neutron_mass_c2 = 939.56536 * MeV;
constexpr double           amu_c2 = 931.494028 * MeV;
constexpr double              amu = amu_c2/c_squared;

//
// permeability of free space mu0    = 2.01334e-16 Mev*(ns*eplus)^2/mm
// permittivity of free space epsil0 = 5.52636e+10 eplus^2/(MeV*mm)
//
constexpr double mu0      = 4*pi*1.e-7 * henry/m;
constexpr double epsilon0 = 1./(c_squared*mu0);

//
// electromagnetic coupling = 1.43996e-12 MeV*mm/(eplus^2)
//
constexpr double elm_coupling           = e_squared/(4*pi*epsilon0);
constexpr double fine_structure_const   = elm_coupling/hbarc;
constexpr double classic_electr_radius  = elm_coupling/electron_mass_c2;
constexpr double electron_Compton_length = hbarc/electron_mass_c2;
constexpr double Bohr_radius = electron_Compton_length/fine_structure_const;

constexpr double alpha_rcl2 = fine_structure_const
                             *classic_electr_radius
                             *classic_electr_radius;

constexpr double twopi_mc2_rcl2 = twopi*electron_mass_c2
                                 *classic_electr_radius
                                 *classic_electr_radius;
//
constexpr double k_Boltzmann = 8.617343e-11 * MeV/kelvin;

//
constexpr double STP_Temperature = 273.15*kelvin;
constexpr double STP_Pressure    = 1.*atmosphere;
constexpr double kGasThreshold   = 10.*mg/cm3;

//
constexpr double universe_mean_density = 1.e-25*g/cm3;

}  // namespace clhep
}  // namespace geantx
