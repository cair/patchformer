#!/bin/bash

seeds="3 11 26"
splits="1 2 3 4"


# DCSWIN Inria Single

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d inria -gpu $gpu -is 512 -n S2 -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# DCSwin Inria Dual

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d inria -gpu $gpu -is 512 -n D3_S2 -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True -dual True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# DCSwin Potsdam Baseline

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d potsdam -gpu $gpu -is 512 -n baseline -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# DCSWIN Potsdam Single

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d potsdam -gpu $gpu -is 512 -n S2 -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# DCSwin Potsdam Dual

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d potsdam -gpu $gpu -is 512 -n D3_S2 -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True -dual True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait