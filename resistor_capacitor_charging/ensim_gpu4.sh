# !/bin/bash

# example 
# saving options: --e (minimal) / {no flag} / --v (maximal)
# width = 0 -> python -u ./simulation_ensemble.py --l $L --w $W --smin 0 --smax 0
# cpu control: --f {desired number of processors}
# --f highly depends on each machine; also check numpy/openblas compilation flags (amd/intel)


# echo "checkpoint 0: $(date +'%H%M%S')"


# L_VALUES=(10 20 30 40 50 60 70)
# W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
# SMIN=0
# SMAX=999

# for L in "${L_VALUES[@]}"; do
#   for W in "${W_VALUES[@]}"; do
#     echo "simulation: (L, W) = ($L, $W)"
#     SECONDS=0


#     python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 15


#     printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
#     sleep 10
#   done
# done


# echo "checkpoint 1: $(date +'%H%M%S')"


echo "checkpoint 0: $(date +'%H%M%S')"


L_VALUES=(22 24 26 28 32 34 36 38)
W_VALUES=(0.8 0.85 0.9 0.95)
SMIN=0
SMAX=332

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0


    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 25


    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 10
  done
done


echo "checkpoint 1: $(date +'%H%M%S')"