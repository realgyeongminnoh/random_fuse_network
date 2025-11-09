#!/usr/bin/env bash

echo "checkpoint: $(date +'%H%M%S')"

L_VALUES=(10 20 30 40 50 60 70 80)
W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=999

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0

    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 18 --cap 100.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 18 --cap 10.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 18 --cap 1.0
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 18 --cap 0.1
    python -u ./simulation_seeds.py --l $L --w $W --smin $SMIN --smax $SMAX --cpu 18 --cap 0.01

    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 1
  done
  echo "checkpoint: $(date +'%H%M%S')"
done

echo "checkpoint: $(date +'%H%M%S')"