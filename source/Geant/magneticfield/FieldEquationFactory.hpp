
#pragma once

// Base types
// #include "Geant/magneticfield/VVectorField.hpp"
#include "Geant/magneticfield/VScalarEquationOfMotion.hpp"

// Concrete Types
// #include "Geant/magneticfield/ScalarUniformMagField.hpp"
#include "Geant/magneticfield/ScalarMagFieldEquation.hpp"

class FieldEquationFactory {
public:
  static const unsigned int Nposmom = 6; // Position 3-vec + Momentum 3-vec

  template <typename FieldType>
  static ScalarMagFieldEquation<FieldType, Nposmom> * // VScalarEquationOfMotion*
  CreateMagEquation(FieldType *field);
  //  Create an equation given a field type

  template <typename FieldType>
  static ScalarMagFieldEquation<FieldType, Nposmom> *CreateMagEquation(FieldType &field);
  //  Similar for a field reference
};

template <typename FieldType>
ScalarMagFieldEquation<FieldType, FieldEquationFactory::Nposmom> * // VScalarEquationOfMotion*
FieldEquationFactory::CreateMagEquation(FieldType *pField)
{
  return new ScalarMagFieldEquation<FieldType, Nposmom>(pField);
}

template <typename FieldType>
ScalarMagFieldEquation<FieldType, FieldEquationFactory::Nposmom> * // VScalarEquationOfMotion*
FieldEquationFactory::CreateMagEquation(FieldType &field)
{
  return new ScalarMagFieldEquation<FieldType, Nposmom>(&field);
}

