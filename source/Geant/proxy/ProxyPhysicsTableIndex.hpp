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
 * @file Geant/proxy/ProxyPhysicsTableType.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//
#pragma once

#include <string>

namespace geantx {

enum ProxyPhysicsTableIndex {
  kNullTableType = -1,          //
  kDEDX_eBrem_eminus,  		// DEDX.eBrem.e-.asc	    
  kDEDX_eBrem_eplus,  		// DEDX.eBrem.e+.asc	    
  kDEDX_eIoni_eminus,  		// DEDX.eIoni.e-.asc	    
  kDEDX_eIoni_eplus,  		// DEDX.eIoni.e+.asc	    
  kInverseRange_eIoni_eminus,  	// InverseRange.eIoni.e-.asc   
  kInverseRange_eIoni_eplus,  	// InverseRange.eIoni.e+.asc   
  kIonisation_eIoni_eminus,  	// Ionisation.eIoni.e-.asc	    
  kIonisation_eIoni_eplus,  	// Ionisation.eIoni.e+.asc	    
  kLambda_compt_gamma,  	// Lambda.compt.gamma.asc	    
  kLambda_conv_gamma,   	// Lambda.conv.gamma.asc	    
  kLambda_eBrem_eminus,  	// Lambda.eBrem.e-.asc	    
  kLambda_eBrem_eplus,  	// Lambda.eBrem.e+.asc	    
  kLambda_eIoni_eminus,  	// Lambda.eIoni.e-.asc	    
  kLambda_eIoni_eplus,  	// Lambda.eIoni.e+.asc	    
  kLambdaMod1_msc_eminus,  	// LambdaMod1.msc.e-.asc	    
  kLambdaMod1_msc_eplus,  	// LambdaMod1.msc.e+.asc	    
  kLambdaPrim_compt_gamma,      // LambdaPrim.compt.gamma.asc  
  kLambdaPrim_phot_gamma,  	// LambdaPrim.phot.gamma.asc   
  kRange_eIoni_eminus,  	// Range.eIoni.e-.asc	    
  kRange_eIoni_eplus, 		// Range.eIoni.e+.asc          
  kNumberPhysicsTable           //     
};

const std::string ProxyPhysicsTableName[kNumberPhysicsTable] = {
  "DEDX.eBrem.e-.asc",
  "DEDX.eBrem.e+.asc",
  "DEDX.eIoni.e-.asc",
  "DEDX.eIoni.e+.asc",
  "InverseRange.eIoni.e-.asc",
  "InverseRange.eIoni.e+.asc",
  "Ionisation.eIoni.e-.asc",
  "Ionisation.eIoni.e+.asc",
  "Lambda.compt.gamma.asc",
  "Lambda.conv.gamma.asc",
  "Lambda.eBrem.e-.asc",
  "Lambda.eBrem.e+.asc",
  "Lambda.eIoni.e-.asc",
  "Lambda.eIoni.e+.asc",
  "LambdaMod1.msc.e-.asc",
  "LambdaMod1.msc.e+.asc",
  "LambdaPrim.compt.gamma.asc",
  "LambdaPrim.phot.gamma.asc",
  "Range.eIoni.e-.asc",
  "Range.eIoni.e+.asc"
};


  //static const char* ProxyPhysicsTableName[ProxyPhysicsTableIndex::kNumberPhysicsTable];
  //static const char* ProxyPhysicsTableName[kNumberPhysicsTable];

} // namespace geantx
