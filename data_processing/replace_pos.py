import os
from nltk.tree import Tree

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)

args = parser.parse_args()

input_pos_dir = os.path.join('./POS', args.dataset)
input_parse_dir = os.path.join('../data/', args.dataset)
output_dir = os.path.join('../data/', args.dataset + '_POS')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for flag in ['train', 'dev', 'test']:
    input_pos_file = os.path.join(input_pos_dir, flag + '.pos')
    input_parse_file = os.path.join(input_parse_dir, flag + '.mrg')
    output_file = os.path.join(output_dir, flag + '.mrg')

    with open(input_pos_file, 'r', encoding='utf8') as f:
        pos_lines = f.readlines()
    with open(input_parse_file, 'r', encoding='utf8') as f:
        parse_lines = f.readlines()
    new_parse = []
    for index, (pos_tags, parse) in enumerate(zip(pos_lines, parse_lines)):
        pos_tags = pos_tags.strip()
        pos_list = pos_tags.split()
        parse = parse.strip()
        parse_tree = Tree.fromstring(parse)
        assert len(pos_list) == len(parse_tree.pos())
        for i, s in enumerate(parse_tree.subtrees(lambda t: t.height() == 2)):
            s.set_label(pos_list[i])
        new_parse.append(' '.join(str(parse_tree).split()))

    with open(output_file, 'w', encoding='utf8') as f:
        for parse in new_parse:
            f.write(parse)
            f.write('\n')
