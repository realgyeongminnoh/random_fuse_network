# !/bin/bash

# example 
# saving options: --e (minimal) / {no flag} / --v (maximal)
# cpu control: --c {desired number of processors}
# width = 0 -> python -u ./simulation_ensemble.py --l $L --w $W --smin 0 --smax 0


echo "checkpoint 0: $(date +'%H%M%S')"


L_VALUES=(10 20 30 40)
W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=2499

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0

    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX

    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 5
  done
done


echo "checkpoint 1: $(date +'%H%M%S')"
sleep 60


L_VALUES=(50 60 70)
W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=2499

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0

    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX

    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 10
  done
done


echo "checkpoint 2: $(date +'%H%M%S')"
sleep 60


L_VALUES=(80 90)
W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=2499

for L in "${L_VALUES[@]}"; do
  for W in "${W_VALUES[@]}"; do
    echo "simulation: (L, W) = ($L, $W)"
    SECONDS=0

    python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX

    printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
    sleep 15
  done
done


echo "checkpoint 3: $(date +'%H%M%S')"
sleep 60


L=100
W_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
SMIN=0
SMAX=2499

for W in "${W_VALUES[@]}"; do
  echo "simulation: (L, W) = ($L, $W)"
  SECONDS=0

  python -u ./simulation_ensemble.py --l $L --w $W --smin $SMIN --smax $SMAX

  printf "elapsed time: %dm %ds\n" $((SECONDS / 60)) $((SECONDS % 60))
  sleep 20
done


echo "checkpoint 4: $(date +'%H%M%S')"
sleep 60