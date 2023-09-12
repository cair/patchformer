#!/bin/bash

seeds="1 7 23"
splits="1 2 3 4"

# DCSwin Inria NewSmall

gpu=0
group="new_small"

for split in $splits; do
  cur_group="${group}_${split}"
  for seed in $seeds; do
    python main.py -m hiera -d inria -gpu $gpu -is 512 -n $group -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -pl True -g $cur_group &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait

# Hiera Inria Single

gpu=0
group="baseline"

for split in $splits; do
  cur_group="${group}_${split}"
  for seed in $seeds; do
    python main.py -m hiera -d inria -gpu $gpu -is 512 -n $group -ms base -lr 0.0001 -e 40 -wb True -split $split -seed $seed -g $cur_group &
    gpu=$((gpu + 1))
    echo Started for gpu $gpu seed $seed split $split
  done
done

wait