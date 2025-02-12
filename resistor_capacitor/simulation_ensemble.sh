# !/bin/bash

# example 
# saving options: --e (minimal) / {no flag} / --v (maximal)
# cpu control: --c {desired number of processors}

L_VALUES=(3 4 5 6 7 8 9 10)
R_VALUES=(2.0)
SMIN=0
SMAX=4999

for L in "${L_VALUES[@]}"; do
  # python -u ./simulation_ensemble.py --l $L --w $R --smin 0 --smax 0 # width=0
  for R in "${R_VALUES[@]}"; do
    python -u ./simulation_ensemble.py --l $L --w $R --smin $SMIN --smax $SMAX --e
  done
done