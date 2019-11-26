#pragma once

#include "Geant/cuts/CutConverter.hpp"

namespace geantx {  
inline namespace GEANT_IMPL_NAMESPACE {

/**
 * @brief   Production threshold converter for positron.
 * @class   CutConverterForPositron
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class CutConverterForPositron : public CutConverter {
public:
  CutConverterForPositron(int numebins = 301, double mincutenergy = 100.0 * geantx::units::eV,
                          double maxcutenergy = 10.0 * geantx::units::GeV);
  virtual ~CutConverterForPositron();

  virtual void Initialise();

protected:
  virtual double ComputeELossOrAbsXsecPerAtom(double zet, double ekin);
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
