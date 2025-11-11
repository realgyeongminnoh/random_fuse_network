#!/usr/bin/env bash

echo "checkpoint: $(date +'%H%M%S')"

L_VALUES=(80)
W_VALUES=(1.0 1.5 1.75 2.0)
SMIN=0
SMAX=65

# L_VALUES=(80)
# W_VALUES=(1.25)
# SMIN=0
# SMAX=65

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0

    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 19 --cap 100.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 19 --cap 10.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 19 --cap 1.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 19 --cap 0.1
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 19 --cap 0.01

    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 1
  done
  echo "checkpoint: $(date +'%H%M%S')"
done

echo "checkpoint: $(date +'%H%M%S')"