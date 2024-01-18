#!/bin/bash
g++ -o3 -std=c++17 -o $1 $2.cpp
cpuprofile=$1.prof dyld_insert_libraries=/usr/local/cellar/gperftools/2.6.3/lib/libprofiler.dylib ./$1
pprof $1 $1.prof > $1.txt
echo "profiling results: $1.txt"