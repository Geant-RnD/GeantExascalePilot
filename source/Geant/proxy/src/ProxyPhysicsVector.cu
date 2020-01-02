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
 * @file Geant/proxy/ProxyPhysicsVector.cu
 * @brief
 */
//===----------------------------------------------------------------------===//
//
#include "Geant/proxy/ProxyPhysicsVector.cuh"


#include <fstream>
#include <iomanip>

namespace geantx {

GEANT_HOST_DEVICE 
ProxyPhysicsVector::ProxyPhysicsVector()
{
  fType = 0;
  fEdgeMin = 0;
  fEdgeMax = 0;
  fNumberOfNodes = 0;
  fUseSpline = false;
  fBaseBin = 0;
  fDeltaBin = 0;
  fIsSecondDerivativeFilled = false;

  fDataVector = nullptr;
  fBinVector = nullptr;
  fSecDerivative = nullptr;
}

GEANT_HOST_DEVICE 
ProxyPhysicsVector::~ProxyPhysicsVector()
{
  Deallocate();
}

GEANT_HOST_DEVICE 
ProxyPhysicsVector::ProxyPhysicsVector(ProxyPhysicsVector& right)
{
  fUseSpline = right.fUseSpline;
  fIsSecondDerivativeFilled = right.fIsSecondDerivativeFilled;
  fNumberOfNodes = right.fNumberOfNodes;
  fEdgeMin = right.fEdgeMin;
  fEdgeMax = right.fEdgeMax;
  fDeltaBin = right.fDeltaBin;

  for (int i = 0; i < fNumberOfNodes; ++i) {
    fDataVector[i] = right.fDataVector[i];
    fBinVector[i] = right.fBinVector[i];
    fSecDerivative[i] = right.fSecDerivative[i];
  }
}

GEANT_HOST
void ProxyPhysicsVector::Allocate(int nsize)
{
  // try
  fDataVector = new double[nsize];
  fBinVector = new double[nsize];
  fSecDerivative = new double[nsize];

  // initialize this vector
  for (int i = 0; i < nsize; ++i) {
    fBinVector[i] = -1;
    fDataVector[i] = 0;
    fSecDerivative[i] = 0;
  }
}

GEANT_HOST_DEVICE 
void ProxyPhysicsVector::Deallocate()
{
  delete [] fBinVector;
  delete [] fDataVector;
  delete [] fSecDerivative;
}

GEANT_HOST
bool ProxyPhysicsVector::Retrieve(std::ifstream& fIn)
{
  // retrieve in ascii mode

  // binning
  fIn >> fEdgeMin >> fEdgeMax >> fNumberOfNodes; 
  if (fIn.fail())  { return false; }
  // contents
  int siz=0;
  fIn >> siz;
  if (fIn.fail() || siz <= 0) { return false; }
 
  //allocate memory
  Allocate(siz);

  double vBin, vData;
 
  for(int i = 0; i < siz ; ++i)
  {
    vBin = 0.;
    vData= 0.;
    fIn >> vBin >> vData;
    if (fIn.fail())  { return false; }
    
    fBinVector[i] = vBin;
    fDataVector[i] = vData;
  }
 
  // to remove any inconsistency 
  fNumberOfNodes = siz;
  fEdgeMin = fBinVector[0];
  fEdgeMax = fBinVector[fNumberOfNodes-1];

  //G4PhysicsLogVector - type 2 : check for InverseRange (type 4)
  fDeltaBin = std::log(fBinVector[1]/fEdgeMin);
  fBaseBin = std::log(fEdgeMin)/fDeltaBin;

  return true;
}

GEANT_HOST_DEVICE
int ProxyPhysicsVector::SizeOfVector()
{
  return 2*(sizeof(bool) + sizeof(int)) + (4 + 3*fNumberOfNodes)*sizeof(double);
}

#ifdef GEANT_CUDA

GEANT_HOST
void ProxyPhysicsVector::Relocate(void *devPtr)
{
  double *d_binVector;
  double *d_dataVector;
  double *d_secDerivative;

  cudaMalloc((void **)&(d_binVector), sizeof(double) * fNumberOfNodes);
  cudaMalloc((void **)&(d_dataVector), sizeof(double) * fNumberOfNodes);
  cudaMalloc((void **)&(d_secDerivative), sizeof(double) * fNumberOfNodes);

  // Copy array contents from host to device.
  cudaMemcpy(d_binVector, fBinVector, sizeof(double) * fNumberOfNodes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dataVector, fDataVector, sizeof(double) * fNumberOfNodes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_secDerivative, fSecDerivative, sizeof(int) * fNumberOfNodes, cudaMemcpyHostToDevice);

  // point to device pointer in host struct.
  fBinVector = d_binVector;
  fDataVector = d_dataVector;
  fSecDerivative = d_secDerivative;

  cudaMemcpy(devPtr, this, SizeOfVector(), cudaMemcpyHostToDevice);
}
#endif

GEANT_HOST_DEVICE
bool ProxyPhysicsVector::SplinePossible()
{
  // Initialise second derivative array. If neighbor energy coincide 
  // or not ordered than spline cannot be applied
  
  for (int i = 0; i < fNumberOfNodes; ++i)
    fSecDerivative[i] = 0.0;
  if (!fUseSpline) {
    return fUseSpline;
  }

  for (int j = 0; j < fNumberOfNodes; ++j) {
    fSecDerivative[j] = 0.0;
    if (j > 0) {
      if (fBinVector[j] - fBinVector[j - 1] <= 0.) {
	fUseSpline = false;
      }
    }
  }
  return fUseSpline;
}

GEANT_HOST_DEVICE
void ProxyPhysicsVector::ComputeSecDerivatives() 
{
  //  A simplified method of computation of second derivatives 
  
  if (!SplinePossible()) {
    return;
  }
  
  if (3 > fNumberOfNodes) // cannot compute derivatives for less than 4 bins
    {
      fUseSpline = false;
      return;
    }
  
  int n = fNumberOfNodes - 1;
  
  for (int i = 1; i < n; ++i) {
    fSecDerivative[i] = 3.0
      * ((fDataVector[i + 1] - fDataVector[i])
	 / (fBinVector[i + 1] - fBinVector[i])
	 - (fDataVector[i] - fDataVector[i - 1])
	 / (fBinVector[i] - fBinVector[i - 1]))
      / (fBinVector[i + 1] - fBinVector[i - 1]);
  }
  fSecDerivative[n] = fSecDerivative[n - 1];
  fSecDerivative[0] = fSecDerivative[1];
}

GEANT_HOST_DEVICE
void ProxyPhysicsVector::ComputeSecondDerivatives(double firstPointDerivative,
					       double endPointDerivative) 
{
  //  A standard method of computation of second derivatives 
  //  First derivatives at the first and the last point should be provided
  //  See for example W.H. Press et al. "Numerical reciptes and C"
  //  Cambridge University Press, 1997.
  
  if (4 > fNumberOfNodes) { // cannot compute derivatives for less than 4 bins
    ComputeSecDerivatives();
    return;
  }
  
  if (!SplinePossible()) {
    return;
  }
  
  int n = fNumberOfNodes - 1;
  
  double* u = new double [n];
  
  double p, sig, un;
  
  u[0] = (6.0 / (fBinVector[1] - fBinVector[0]))
    * ((fDataVector[1] - fDataVector[0]) / (fBinVector[1] - fBinVector[0])
       - firstPointDerivative);
  
  fSecDerivative[0] = -0.5;
  
  // Decomposition loop for tridiagonal algorithm. fSecDerivative[i]
  // and u[i] are used for temporary storage of the decomposed factors.
  
  for (int i = 1; i < n; ++i) {
    sig = (fBinVector[i] - fBinVector[i - 1])
      / (fBinVector[i + 1] - fBinVector[i - 1]);
    p = sig * (fSecDerivative[i - 1]) + 2.0;
    fSecDerivative[i] = (sig - 1.0) / p;
    u[i] = (fDataVector[i + 1] - fDataVector[i])
      / (fBinVector[i + 1] - fBinVector[i])
      - (fDataVector[i] - fDataVector[i - 1])
      / (fBinVector[i] - fBinVector[i - 1]);
    u[i] = 6.0 * u[i] / (fBinVector[i + 1] - fBinVector[i - 1])
      - sig * u[i - 1] / p;
  }
  
  sig = (fBinVector[n - 1] - fBinVector[n - 2])
    / (fBinVector[n] - fBinVector[n - 2]);
  p = sig * fSecDerivative[n - 2] + 2.0;
  un = (6.0 / (fBinVector[n] - fBinVector[n - 1]))
    * (endPointDerivative
       - (fDataVector[n] - fDataVector[n - 1])
       / (fBinVector[n] - fBinVector[n - 1])) - u[n - 1] / p;
  fSecDerivative[n] = un / (fSecDerivative[n - 1] + 2.0);
  
  // The back-substitution loop for the triagonal algorithm of solving
  // a linear system of equations.
  
  for (int k = n - 1; k > 0; --k) {
    fSecDerivative[k] *= (fSecDerivative[k + 1]
			 - u[k] * (fBinVector[k + 1] - fBinVector[k - 1])
			 / (fBinVector[k + 1] - fBinVector[k]));
  }
  fSecDerivative[0] = 0.5 * (u[0] - fSecDerivative[1]);
  
  delete u;
}

GEANT_HOST_DEVICE
void ProxyPhysicsVector::FillSecondDerivatives()
{
  // Computation of second derivatives using "Not-a-knot" endpoint conditions
  // B.I. Kvasov "Methods of shape-preserving spline approximation"
  // World Scientific, 2000
  
  if (5 > fNumberOfNodes) // cannot compute derivatives for less than 4 points
    {
      ComputeSecDerivatives();
      return;
    }
  
  if (!SplinePossible()) {
    return;
  }
  
  int n = fNumberOfNodes - 1;
  
  double* u = new double [n];
  
  double p, sig;
  
  u[1] = ((fDataVector[2] - fDataVector[1]) / (fBinVector[2] - fBinVector[1])
	  - (fDataVector[1] - fDataVector[0]) / (fBinVector[1] - fBinVector[0]));
  u[1] = 6.0 * u[1] * (fBinVector[2] - fBinVector[1])
    / ((fBinVector[2] - fBinVector[0]) * (fBinVector[2] - fBinVector[0]));
  
  // Decomposition loop for tridiagonal algorithm. fSecDerivative[i]
  // and u[i] are used for temporary storage of the decomposed factors.
  
  fSecDerivative[1] = (2.0 * fBinVector[1] - fBinVector[0] - fBinVector[2])
    / (2.0 * fBinVector[2] - fBinVector[0] - fBinVector[1]);
  
  for (int i = 2; i < n - 1; ++i) {
    sig = (fBinVector[i] - fBinVector[i - 1])
      / (fBinVector[i + 1] - fBinVector[i - 1]);
    p = sig * fSecDerivative[i - 1] + 2.0;
    fSecDerivative[i] = (sig - 1.0) / p;
    u[i] = (fDataVector[i + 1] - fDataVector[i])
      / (fBinVector[i + 1] - fBinVector[i])
      - (fDataVector[i] - fDataVector[i - 1])
      / (fBinVector[i] - fBinVector[i - 1]);
    u[i] = (6.0 * u[i] / (fBinVector[i + 1] - fBinVector[i - 1]))
      - sig * u[i - 1] / p;
  }
  
  sig = (fBinVector[n - 1] - fBinVector[n - 2])
    / (fBinVector[n] - fBinVector[n - 2]);
  p = sig * fSecDerivative[n - 3] + 2.0;
  u[n - 1] = (fDataVector[n] - fDataVector[n - 1])
    / (fBinVector[n] - fBinVector[n - 1])
    - (fDataVector[n - 1] - fDataVector[n - 2])
    / (fBinVector[n - 1] - fBinVector[n - 2]);
  u[n - 1] = 6.0 * sig * u[n - 1] / (fBinVector[n] - fBinVector[n - 2])
    - (2.0 * sig - 1.0) * u[n - 2] / p;
  
  p = (1.0 + sig) + (2.0 * sig - 1.0) * fSecDerivative[n - 2];
  fSecDerivative[n - 1] = u[n - 1] / p;
  
  // The back-substitution loop for the triagonal algorithm of solving
  // a linear system of equations.
  
  for (int k = n - 2; k > 1; --k) {
    fSecDerivative[k] *= (fSecDerivative[k + 1]
			 - u[k] * (fBinVector[k + 1] - fBinVector[k - 1])
			 / (fBinVector[k + 1] - fBinVector[k]));
  }
  fSecDerivative[n] = (fSecDerivative[n - 1]
		      - (1.0 - sig) * fSecDerivative[n - 2]) / sig;
  sig = 1.0 - ((fBinVector[2] - fBinVector[1]) / (fBinVector[2] - fBinVector[0]));
  fSecDerivative[1] *= (fSecDerivative[2] - u[1] / (1.0 - sig));
  fSecDerivative[0] = (fSecDerivative[1] - sig * fSecDerivative[2])
    / (1.0 - sig);
  
  fIsSecondDerivativeFilled = true;

  delete u;
}

GEANT_HOST_DEVICE
int ProxyPhysicsVector::FindBinLocation(double energy)
{
  int bin = log(energy)/fDeltaBin - fBaseBin;
  if(bin > 0 && energy < fBinVector[bin]) { --bin; }
  return bin;
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::SplineInterpolation(double energy, int lastBin)
{
  // Spline interpolation is used to get the value. If the give energy
  // is in the highest bin, no interpolation will be Done. Because 
  // there is an extra bin hidden from a user at locBin=numberOfBin, 
  // the following interpolation is valid even the current locBin=
  // numberOfBin-1. 
  
  //  if(0 == fSecDerivative.size() ) { FillSecondDerivatives(); }
  if (!fIsSecondDerivativeFilled) {
    FillSecondDerivatives();
  }
  
  // check bin value
  double x1 = fBinVector[lastBin];
  double x2 = fBinVector[lastBin + 1];
  double delta = x2 - x1;
  
  double a = (x2 - energy) / delta;
  double b = (energy - x1) / delta;
  
  // Final evaluation of cubic spline polynomial for return   
  double y1 = fDataVector[lastBin];
  double y2 = fDataVector[lastBin + 1];
  

  double res = a * y1 + b * y2
    + ((a * a * a - a) * fSecDerivative[lastBin]
       + (b * b * b - b) * fSecDerivative[lastBin + 1]) * delta
    * delta / 6.0;

  return res;
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::LinearInterpolation(double energy, int lastBin)
{
  // Linear interpolation is used to get the value. If the give energy
  // is in the highest bin, no interpolation will be Done. Because 
  // there is an extra bin hidden from a user at locBin=numberOfBin, 
  // the following interpolation is valid even the current locBin=
  // numberOfBin-1. 
  
  double intplFactor = (energy - fBinVector[lastBin])
    / (fBinVector[lastBin + 1] - fBinVector[lastBin]); // Interpol. factor
  
  return fDataVector[lastBin]
    + (fDataVector[lastBin + 1] - fDataVector[lastBin]) * intplFactor;
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::Interpolation(double energy, int lastBin)
{
  double value = 0;
  if (fUseSpline) {
    value = SplineInterpolation(energy, lastBin);
  } else {
    value = LinearInterpolation(energy, lastBin);
  }
  return value;
}

GEANT_HOST_DEVICE
void ProxyPhysicsVector::SetSpline(bool val)
{
  fUseSpline = val;
  if (!fIsSecondDerivativeFilled) {
    FillSecondDerivatives();
    fIsSecondDerivativeFilled = true;
  }
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::Value(double energy)
{
  double value = 0.0;
  
  if (energy <= fEdgeMin) {
    value = fDataVector[0];
  } else if (energy >= fEdgeMax) {
    value = fDataVector[fNumberOfNodes - 1];
  } else {
    int bin = FindBinLocation(energy);
    value = Interpolation(energy, bin);
  }
  return value;
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::Energy(size_t binNumber)
{
  return fBinVector[binNumber];
}

GEANT_HOST_DEVICE
double ProxyPhysicsVector::InvRangeValue(double energy) 
{
  double value = 0.0;
  
  if (energy <= fEdgeMin) {
    value = fDataVector[0];
  } else if (energy >= fEdgeMax) {
    value = fDataVector[fNumberOfNodes - 1];
  } else {
    int bin = InvRangeFindBinLocation(energy);
    value = Interpolation(energy, bin);
  }
  return value;
}

GEANT_HOST_DEVICE
int ProxyPhysicsVector::InvRangeFindBinLocation(double energy) 
{
  int n1 = 0;
  int n2 = fNumberOfNodes/2;
  int n3 = fNumberOfNodes - 1;
  while (n1 != n3 - 1) {
    if (energy > fBinVector[n2]) { 
      n1 = n2; 
    }
    else { 
      n3 = n2; 
    }
    n2 = n1 + (n3 - n1 + 1)/2;
  }
  
  return (int)n1;
}

} // namespace geantx
