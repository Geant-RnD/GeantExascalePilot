#!/bin/bash

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

