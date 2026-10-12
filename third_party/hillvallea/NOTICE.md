# HillVallEA

HillVallEA (Maree, Alderliesten, Thierens and Bosman) is not included in this repository.
`build.sh` downloads the authors' code (https://github.com/SCMaree/HillVallEA, commit 1a68ab7,
GPL-3.0), applies `seed.patch` and builds the shared library used by `src/aumo/algorithms/hillvallea.py`.

`seed.patch` modifies GPL-3.0 code and `hve_bridge.cpp` is compiled against it; both files are
distributed under the GPL-3.0, unlike the rest of this repository (MIT).
