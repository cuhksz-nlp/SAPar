#!/bin/bash

python build_atb.py --atb_part1=./LDC03T06 --atb_part2=./LDC04T02 --atb_part3=./LDC05T20 --split_dir=./atb_stanford_split --output_dir=../data/ATB/tmp

python strip_functional.py < ../data/ATB/tmp/train.raw.mrg | python ensure_top.py > ../data/ATB/tmp/train.cleaned.mrg
python strip_functional.py < ../data/ATB/tmp/dev.raw.mrg | python ensure_top.py > ../data/ATB/tmp/dev.cleaned.mrg
python strip_functional.py < ../data/ATB/tmp/test.raw.mrg | python ensure_top.py > ../data/ATB/tmp/test.cleaned.mrg

python atb_convert.py --input_dir=../data/ATB/tmp/ --output_dir=../data/ATB/

rm -rf ../data/ATB/tmp

# The following command line generate data files with predicted POS tags
python replace_pos.py --dataset=ATB
