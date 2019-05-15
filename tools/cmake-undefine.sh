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
# @brief Find existing variable prefix by the script's parameters in
#        CMakeCache.txt and generate the list of corresponding -U
#
#===----------------------------------------------------------------------===//

STR=""
for i in $@
do
    _tmp=$(grep "^${i}" CMakeCache.txt | grep -v 'ADVANCED' | sed 's/:/ /g' | awk '{print $1}')
    for j in ${_tmp}
    do
        if [ ! -z "${STR}" ]; then
            STR="${STR} -U${j}"
        else
            STR="-U${j}"
        fi
    done
done

if [ ! -z "${STR}" ]; then
    echo "${STR}"
fi

