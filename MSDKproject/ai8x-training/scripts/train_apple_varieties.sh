#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --batch-size 8 --model ai85apple_discrimination --dataset AppleSpectra --data ./data/Apple_varieties --confusion --device MAX78000 "$@"
