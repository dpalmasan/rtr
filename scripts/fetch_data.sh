#!/bin/sh
# Downloads the benchmark data from the organisers' repositories into data/ (not redistributed here).
#   CEC 2013 niching suite: github.com/mikeagn/CEC2013 (python3/data, BSD licence)
#   CEC 2022 dynamic multimodal suite: github.com/MiLabHITSZ/dmmops (problem/Data, problem/phi.mat;
#     "for academic purpose only", see the organisers' repository)
set -e
cd "$(dirname "$0")/.."
mkdir -p data/cec2013 data/dmmop
C=https://raw.githubusercontent.com/mikeagn/CEC2013/5ffda55a9a3c4424297b816e6593e6aa1cbd8aea/python3/data
for f in CF3_M_D2 CF3_M_D3 CF3_M_D5 CF3_M_D10 CF3_M_D20 CF4_M_D2 CF4_M_D3 CF4_M_D5 CF4_M_D10 CF4_M_D20 optima; do
  curl -fsSL -o "data/cec2013/$f.dat" "$C/$f.dat"
done
curl -fsSL -o data/cec2013/LICENSE.txt https://raw.githubusercontent.com/mikeagn/CEC2013/5ffda55a9a3c4424297b816e6593e6aa1cbd8aea/LICENSE.txt || true
D=https://raw.githubusercontent.com/MiLabHITSZ/dmmops/ddc560c8c533b650c2a2e29cadc1e321b3e1abd8/problem
for f in CF3_M_D2 CF3_M_D3 CF3_M_D5 CF3_M_D10 CF3_M_D20 CF4_M_D2 CF4_M_D3 CF4_M_D5 CF4_M_D10 CF4_M_D20 F1X optima; do
  curl -fsSL -o "data/dmmop/$f.mat" "$D/Data/$f.mat"
done
curl -fsSL -o data/dmmop/phi.mat "$D/phi.mat"
echo "data ready in $(pwd)/data"
