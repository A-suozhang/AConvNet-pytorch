#!/bin/bash

#python generate_dataset.py --is_train=True --use_phase=True --chip_size=100 --patch_size=94 --use_phase=True --dataset=soc 
#python generate_dataset.py --is_train=False --use_phase=True --chip_size=128 --patch_size=128  --use_phase=True --dataset=soc
python generate_dataset.py --is_train=True --use_phase=True --chip_size=100 --patch_size=94 --use_phase=True --dataset=eoc-1-t72-a64 
python generate_dataset.py --is_train=False --use_phase=True --chip_size=128 --patch_size=128  --use_phase=True --dataset=eoc-1-t72-a64
