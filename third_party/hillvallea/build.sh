#!/bin/sh
# Fetches HillVallEA at the GECCO 2019 commit, applies seed.patch and builds libhve.so here.
set -e
cd "$(dirname "$0")"
COMMIT=1a68ab7
[ -d src ] || git clone -q https://github.com/SCMaree/HillVallEA src
git -C src checkout -q "$COMMIT"
git -C src checkout -q -- .
patch -s -d src -p1 < seed.patch
g++ -O3 -std=c++11 -fPIC -shared -I src -o libhve.so hve_bridge.cpp src/HillVallEA/*.cpp
echo "built $(pwd)/libhve.so"
