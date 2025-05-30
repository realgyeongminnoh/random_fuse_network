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


# echo "checkpoint 0: $(date +'%H%M%S')"


# L_VALUES=(10 20 22 24 26 28 30 32 34 36 38 40 50 60 70)
# W_VALUES=(1.05 1.1 1.15 1.2)
# SMIN=667
# SMAX=999

# for L in "${L_VALUES[@]}"; do
#   for W in "${W_VALUES[@]}"; do
#     echo "simulation: (L, W) = ($L, $W)"
#     SECONDS=0


#     python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 10


#     printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
#     sleep 5
#   done
# done


# echo "checkpoint 1: $(date +'%H%M%S')"



echo "checkpoint 0: $(date +'%H%M%S')"


L_VALUES=(10 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 60 70)
W_VALUES=(0.2 0.3 0.35)
SMIN=1334
SMAX=1999

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0


    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 10


    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 5
  done
done


echo "checkpoint 1: $(date +'%H%M%S')"


L_VALUES=(42 44 46 48)
W_VALUES=(0.25)
SMIN=1334
SMAX=1999

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0


    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 10


    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 5
  done
done


echo "checkpoint 2: $(date +'%H%M%S')"


L_VALUES=(10 20 22 24 26 28 30 32 34 36 38 40 50 60 70 )
W_VALUES=(0.25)
SMIN=1667
SMAX=1999

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0


    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX --f 10


    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 5
  done
done


echo "checkpoint 3: $(date +'%H%M%S')"