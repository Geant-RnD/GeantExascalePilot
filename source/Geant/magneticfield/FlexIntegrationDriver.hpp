//
//  FlexIntegrationDriver.h
//  ALL_BUILD
//
//  Created by japost on 15.12.17.
//

#pragma once

class ScalarFieldTrack;
struct FieldTrack;

//  Attempt to provide a method that works for a single track using Flex/Simple Integration Driver
#define EXTEND_SINGLE 1

class FlexIntegrationDriver {
public:
  /**
    // Scalar method
    virtual
    bool  AccurateAdvance( const ScalarFieldTrack& y_current,
                          double  hstep,
                          double  eps, //same             // Requested y_err/hstep
                          ScalarFieldTrack& yOutput,
                          double  hinitial=0.0) = 0;
   **/

  // Method for array / vector
  virtual void AccurateAdvance(const FieldTrack yInput[], const double hstep[], const double charge[], double epsilon,
                               FieldTrack yOutput[], int nTracks, bool succeeded[]) const = 0;
  // Drive Runge-Kutta integration of ODE for several tracks (ntracks)
  // with starting values yInput, from current 's'=0 to s=h with variable
  // stepsize to control error, so that it is bounded by the relative
  // accuracy eps.  On output yOutput is value at end of interval.
  // The concept is similar to the odeint routine from NRC 2nd edition p.721

#ifdef EXTEND_SINGLE
  /** @brief Single track variant - Experimental (but planned for future use.)  */
  virtual void AccurateAdvance(const FieldTrack &yInput, const double hstep, const double charge, double epsilon,
                               FieldTrack &yOutput, bool succeeded) const = 0;
#endif
};
