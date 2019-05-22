#!/bin/bash

#===------------------ GeantX --------------------------------------------===//
#
# Geant Exascale Pilot
#
# For the licensing terms see LICENSE file.
# For the list of contributors see CREDITS file.
# Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
#===----------------------------------------------------------------------===//
#
# @file
# @brief This is running a clang-format test
#        by doing a filtering step and then analysing
#        the result of applying ./scripts/clang-format-and-fix-macros.sh
#
#===----------------------------------------------------------------------===//

# Originated in the GeantV project.

./scripts/clang-format-apply.sh
res=$?

if [ $res -eq 0 ] ; then 
   # cleanup changes in git
   git reset HEAD --hard
fi

exit $res
