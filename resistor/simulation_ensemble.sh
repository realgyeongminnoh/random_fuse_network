# !/bin/bash

# example 
# saving options: --a (minimal) / {no flag} / --v (maximal)
# cpu control: --c {desired number of processors}

L_VALUES=(3 4 13 30)
R_VALUES=(0.25 0.75 1.25 1.75 2.0)
SMIN=1000
SMAX=1999

for L in "${L_VALUES[@]}"; do
  # python -u ./simulation_ensemble.py --l $L --w $R --smin 0 --smax 0 # width=0
  for R in "${R_VALUES[@]}"; do
    python -u ./simulation_ensemble.py --l $L --w $R --smin $SMIN --smax $SMAX --v
  done
done