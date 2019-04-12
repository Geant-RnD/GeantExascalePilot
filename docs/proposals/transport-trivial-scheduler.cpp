/*
 PreStep; scheduler set the track as starting the step

 FastSimulation: shortcut.

 ComputeIntLS: normal (non msc) physics step limit computation.

 GeomQueryStage: Find next boundary

 PrePropagation: mscProc->AlongStepLimitationLength(track, td);

 Propagation: integration of the equation of motion

 PostPropagation: MSC along step, increase time, decrease interaction length.

 AlongStep: continuous processes.

 PostStep:  eg. produces the secondaries for an in-flight hadronic process (One of the stage that can generate secondaries)

 AtRest: eg. sample the target isotope and produces the secondaries for an at-rest hadronic process

 SteppingAction: User actions.
 */

// this is needed if the threads can share work, otherwise none, error and
// stop are the same.
// enum class FeederResult : char { kNone, kError, kWork, kStop };


// RunManager: (one per process)
// configuration, pointers to User object ('App'), geometry (or
// geometry constructor), physics list, MCTruth, factories (TaskData)
// coordination:
//   propagator, event in flight, priority events, task id generator,
//   end marker, thread lis.

// Propagator: (one per numa-node/sequestered-memory-area)
// Global stats (nsteps, ntransported, etc.)
// Some repetition of the RunManager members.
// collection of stages

// EventServer:
//  Abstraction over the primary particles providers (GunParticle or experiment
//  way of generating primaries).

bool EventLoop() {
   TaskData taskData; // or get from factory.
   Propagator propatgor; // Needs config info.
   TrackCollection tracks;

   while (GetPrimaries(tracks, taskData)) {
      Steppingloop(tracks, taskData);
   }
}

bool SteppingLoop(TrackCollection &trackCollection, TaskData &taskData) {

   // See for alternative equivalent implementaiton https://godbolt.org/z/fpPEiC
   while(!trackCollection.empty())
   {
      TrackRef_t t = trackCollection.back();
      trackCollection.pop_back();

      while (1) // Run until the particle is dead.
      {
         if (! BeginStage(t, trackCollection, taskData) ) break;
         if (! ComputeIntLS(t, trackCollection, taskData) ) break;
         if (! GeomQuery(t, trackCollection, taskData) ) break;
         if (! PrePropagation(t, trackCollection, taskData) ) break;
         if (! Propagation(t, trackCollection, taskData) ) break;
         if (! PostPropagation(t, trackCollection, taskData) ) break;
         if (! AlongStep(t, trackCollection, taskData) ) break;
         if (! PostStep(t, trackCollection, taskData) ) break;
         if (! AtRest(t, trackCollection, taskData) ) break;
         if (! UserSteppingAction(t, trackCollection, taskData) ) break;
      }
      // here GeantV has an opportunity to rebalance and bring in new tracks.
   }
}

/*

 skip options:
 if (!dead) doit()
 or
 (inside doit) if (!dead) push [To 'right' stage buffer]
 next.doit( pop );
 or
 if (dead) continue;
    while(Track *t = trackCollection.pop()) {
      // How to 'skip' stage.
      // How to handle secondaries
      //   GeantV add them to a (semantically) temporary buffer
      //   Then distribute over the stage buffer (with 0th buffer being special)

      while(PreStep(t, trackCollection, taskData) &&
            ComputeIntLS(t, trackCollection, taskData) &&
            GeomQuery(t, trackCollection, taskData) &&
            PrePropagation(t, trackCollection, taskData) &&
            Propagation(t, trackCollection, taskData) &&
            PostPropagation(t, trackCollection, taskData) &&
            AlongStep(t, trackCollection, taskData) &&
            PostStep(t, trackCollection, taskData) &&
            AtRest(t, trackCollection, taskData) &&
            UserSteppingAction(t, trackCollection, taskData) ) {
         DEBUG_PRINT("main loop: will do one more step for track %s\n",t.GetName());
      }

      if (! PreStep(t, trackCollection, taskData) ) continue;
      if (! ComputeIntLS(t, trackCollection, taskData) ) continue;
      if (! GeomQuery(t, trackCollection, taskData) ) continue;
      if (! PrePropagation(t, trackCollection, taskData) ) continue;
      if (! Propagation(t, trackCollection, taskData) ) continue;
      if (! PostPropagation(t, trackCollection, taskData) ) continue;
      if (! AlongStep(t, trackCollection, taskData) ) continue;
      if (! PostStep(t, trackCollection, taskData) ) continue;
      if (! AtRest(t, trackCollection, taskData) ) continue;
      if (! UserSteppingAction(t, trackCollection, taskData) ) continue;


      PreStep(t, trackCollection, taskData);
      if (t.IsDead()) continue
      ComputeIntLS(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      GeomQuery(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      PrePropagation(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      Propagation(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      PostPropagation(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      AlongStep(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      PostStep(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      AtRest(t, trackCollection, taskData);
      if (t.IsDead()) continue;
      UserSteppingAction(t, trackCollection, taskData);
      if (!t.IsDead()) push();
   }

  */

