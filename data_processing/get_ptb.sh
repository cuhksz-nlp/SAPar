#!/bin/bash

python build_ptb.py --data_home=./LDC99T42 --output_dir=../data/PTB/tmp

python strip_functional.py < ../data/PTB/tmp/train.ori.mrg | python ensure_top.py | sed 's/PRT|ADVP/PRT/g' > ../data/PTB/train.mrg
python strip_functional.py < ../data/PTB/tmp/dev.ori.mrg | python ensure_top.py > ../data/PTB/dev.mrg
python strip_functional.py < ../data/PTB/tmp/test.ori.mrg | python ensure_top.py > ../data/PTB/test.mrg

rm -rf ../data/PTB/tmp

# The following command line generate data files with predicted POS tags
python replace_pos.py --dataset=PTB
