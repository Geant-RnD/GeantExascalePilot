//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/random/MRG.hpp
 * @brief NameSpace for the MRG32k3a class based on RngStream.h(cpp)  

  RngStream is a class generating multiple streams of random numbers created
  by Prof. Pierre L'Ecuyer, University of Montreal (lecuyer@iro.umontreal.ca
  Original source codes of RngStream.h(cpp) is available at                 
  http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/                 
                                                                            
  Relevant articles in which MRG32k3a and the package with multiple streams 
  were proposed:                                                            
                                                                            
  P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random 
  Number Generators'', Operations Research, 47, 1 (1999), 159--164.         
                                                                            
  P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton, ``An Objected-Oriented
  Random-Number Package with Many Long Streams and Substreams'', Operations 
  Research, 50, 6 (2002), 1073--1075             
 */
//===----------------------------------------------------------------------===//

#pragma once

namespace geantx {
namespace MRG {

constexpr auto ndim = 3;
constexpr auto vsize = 6;
constexpr auto slength = 127; // stream length in powers of two: 2^127

constexpr double m1 = 4294967087.0;
constexpr double m2 = 4294944443.0;
constexpr double norm = 1.0 / (m1 + 1.0);
constexpr double a12 = 1403580.0;
constexpr double a13n = 810728.0;
constexpr double a21 = 527612.0;
constexpr double a23n = 1370589.0;
constexpr double two17 = 131072.0;
constexpr double two53 = 9007199254740992.0;
constexpr double fact = 5.9604644775390625e-8; // 1 / 2^24

// The following are the transition matrices of the two MRG components
// (in matrix form), raised to the powers -1, 1, 2^76, and 2^127, resp.

constexpr double A1p76[ndim][ndim] = {
    {82758667.0, 1871391091.0, 4127413238.0},
    {3672831523.0, 69195019.0, 1871391091.0},
    {3672091415.0, 3528743235.0, 69195019.0}};

constexpr double A2p76[ndim][ndim] = {
    {1511326704.0, 3759209742.0, 1610795712.0},
    {4292754251.0, 1511326704.0, 3889917532.0},
    {3859662829.0, 4292754251.0, 3708466080.0}};

constexpr double A1p127[ndim][ndim] = {
    {2427906178.0, 3580155704.0, 949770784.0},
    {226153695.0, 1230515664.0, 3580155704.0},
    {1988835001.0, 986791581.0, 1230515664.0}};
constexpr double A2p127[ndim][ndim] = {
    {1464411153.0, 277697599.0, 1610723613.0},
    {32183930.0, 1464411153.0, 1022607788.0},
    {2824425944.0, 32183930.0, 2093834863.0}};

constexpr double A1p0[MRG::ndim][MRG::ndim] = {
    {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-810728.0, 1403580.0, 0.0}};

constexpr double A2p0[MRG::ndim][MRG::ndim] = {
    {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-1370589.0, 0.0, 527612.0}};

// Inverse of A1p0
constexpr double InvA1[MRG::ndim][MRG::ndim] = {
    {184888585.0, 0.0, 1945170933.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

// Inverse of A2p0
constexpr double InvA2[MRG::ndim][MRG::ndim] = {
    {0.0, 360363334.0, 4225571728.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

// parameters from curand_MRG32k3a
constexpr double rh1 = 2.3283065498378290e-010;  /* (1.0 / m1)__hi */
constexpr double rl1 = -1.7354913086174288e-026; /* (1.0 / m1)__lo */
constexpr double rh2 = 2.3283188252407387e-010;  /* (1.0 / m2)__hi */
constexpr double rl2 = 2.4081018096503646e-026;  /* (1.0 / m2)__lo */

} // namespace MRG
} // namespace geantx
