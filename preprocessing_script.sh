#!/bin/bash

SOURCEDIR=./
cd $SOURCEDIR

data_dir="path/to/beam"
mode='test'

################################################
python preprocessing/preprocess_beam.py --data_dir ${data_dir} --removeLand --removeIrrelevant --utm
python preprocessing/split_data_bulk.py --input_dir ${data_dir} --mode ${mode}