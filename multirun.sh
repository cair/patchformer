#!/bin/bash

seeds="3 11 26"
splits="1 2 3 4"

# Dcswin Inria New

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d inria -gpu $gpu -is 512 -n new_conv_dice -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# Hiera Inria Single

gpu=0

for seed in $seeds; do
  for split in $splits; do
    python main.py -m dcswin -d potsdam -gpu $gpu -is 512 -n new_conv_dice -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait
