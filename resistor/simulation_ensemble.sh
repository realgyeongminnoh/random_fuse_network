# !/bin/bash

# example 
# saving options: --a (minimal) / {no flag} / --v (maximal)
# cpu control: --c {desired number of processors}

L_VALUES=(10 20 30 40 50)
R_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=999

for L in "${L_VALUES[@]}"; do
  # python -u ./simulation_parallel_ensemble.py --l $L --w $R --smin 0 --smax 0 # width=0
  for R in "${R_VALUES[@]}"; do
    python -u ./simulation_parallel_ensemble.py --l $L --w $R --smin $SMIN --smax $SMAX 
  done
done