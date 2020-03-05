//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
//
/**
 * @file Geant/proxy/src/ProxyElementUtil.cu
 * @brief
 */
//===----------------------------------------------------------------------===//

#include "Geant/proxy/ProxyElementUtils.cuh"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"

namespace geantx {
namespace ProxyElementUtils {

GEANT_HOST_DEVICE
double ComputeCoulombFactor(double fZeff)
{
  //  Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)

  double k1 = 0.0083 , k2 = 0.20206 ,k3 = 0.0020 , k4 = 0.0369 ;

  double az1 = clhep::fine_structure_const*fZeff;
  double az2 = az1 * az1;
  double az4 = az2 * az2;

  double fCoulomb = (k1*az4 + k2 + 1./(1.+az2))*az2 - (k3*az4 + k4)*az4;

  return fCoulomb;
}

GEANT_HOST_DEVICE
double ComputeLradTsaiFactor(double fZeff)
{
  //  Compute Tsai's Expression for the Radiation Length
  //  (Phys Rev. D50 3-1 (1994) page 1254)

  double Lrad_light[]  = {5.31  , 4.79  , 4.74 ,  4.71} ;
  double Lprad_light[] = {6.144 , 5.621 , 5.805 , 5.924} ;
  
  const double logZ3 = log(fZeff)/3.;

  double Lrad, Lprad;
  int iz = (int)(fZeff+0.5) - 1 ;

  if (iz <= 3) { Lrad = Lrad_light[iz] ;  Lprad = Lprad_light[iz] ; }
  else { Lrad = log(184.15) - logZ3 ; Lprad = log(1194.) - 2*logZ3;}

  double fCoulomb = ComputeCoulombFactor(fZeff);

  double fRadTsai = 4*clhep::alpha_rcl2*fZeff*(fZeff*(Lrad-fCoulomb) + Lprad); 

  return fRadTsai;
}

} // namespace ProxyElementUtils
} // namespace geantx
