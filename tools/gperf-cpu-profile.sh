#!/bin/bash -e

: ${N:=0}
: ${CPUPROFILE:=""}
: ${CPUPROFBASE:=gperf.cpu.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=500}
: ${INTERACTIVE:=0}

while [ -z "${CPUPROFILE}" ]
do
    TEST_FILE=${CPUPROFBASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        CPUPROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

echo -e "\n\t--> Outputting profile to '${CPUPROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; rm -f ${CPUPROFILE}; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

# configure pre-loading of profiler library
LIBS=$(find ${PWD} -type f | egrep '\.so|\.dylib' | egrep -v 'vdt|\.o$|\.cmake$|\.txt$|\.sh$|\.a$|\.dSYM')
LIBS=$(echo ${LIBS} | sed 's/ /:/g')
if [ "$(uname)" = "Darwin" ]; then
    for i in $(otool -L ${1} | egrep 'tcmalloc|profiler' | awk '{print $1}')
    do
        if [ ! -d ${i} ]; then
            LIBS=${LIBS}:${i}
        fi
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        export DYLD_INSERT_LIBRARIES=${LIBS}
        echo "DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}"
    fi
    unset LIBS
else
    for i in $(ldd ${1} | egrep 'tcmalloc|profiler' | awk '{print $(NF-1)}')
    do
        if [ ! -d ${i} ]; then
            LIBS=${LIBS}:${i}
        fi
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export LD_PRELOAD=${LIBS}
        echo "LD_PRELOAD=${LD_PRELOAD}"
    fi
    unset LIBS
fi

export MALLOCSTATS
export CPUPROFILE_FREQUENCY
# run the application
eval CPUPROFILE=${CPUPROFILE} $@ | tee ${CPUPROFILE}.log

# generate the results
EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
if [ -f "${CPUPROFILE}" ]; then
    : ${PPROF:=$(which google-pprof)}
    : ${PPROF:=$(which pprof)}
    ADD_LIBS=""
    for i in *.${EXT}
    do
        ADD_LIBS="${ADD_LIBS} --add_lib=${i}"
    done
    if [ -n "${PPROF}" ]; then
        eval ${PPROF} --text ${ADD_LIBS} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.txt
        eval ${PPROF} --text --cum ${ADD_LIBS} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.cum.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            eval ${PPROF} ${ADD_LIBS} ${PPROF_ARGS} ${1} ${CPUPROFILE}
        fi
    fi
fi
