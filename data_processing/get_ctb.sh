#!/bin/bash

python build_ctb.py --data_home=./LDC05T01 --output_dir=../data/CTB/tmp

python strip_functional.py < ../data/CTB/tmp/train.ori.mrg | python ensure_top.py > ../data/CTB/tmp/train.raw.mrg
python strip_functional.py < ../data/CTB/tmp/dev.ori.mrg | python ensure_top.py > ../data/CTB/tmp/dev.raw.mrg
python strip_functional.py < ../data/CTB/tmp/test.ori.mrg | python ensure_top.py > ../data/CTB/tmp/test.raw.mrg

python ctb_clean.py --input_dir=../data/CTB/tmp --output_dir=../data/CTB

rm -rf ../data/CTB/tmp

# The following command line generate data files with predicted POS tags
python replace_pos.py --dataset=CTB
