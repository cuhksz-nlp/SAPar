import os
from os import path
from nltk.tree import Tree
import argparse

# MOD/ATB1_v2_0/data/treebank/with-vowel

parser = argparse.ArgumentParser()
parser.add_argument("--atb_part1", required=True)
parser.add_argument("--atb_part2", required=True)
parser.add_argument("--atb_part3", required=True)
parser.add_argument("--split_dir", required=True)
parser.add_argument("--output_dir", required=True, type=str)

args = parser.parse_args()

# part1_home = './LDC03T06/'
# part2_home = './LDC04T02/'
# part3_home = './LDC05T20/'

part1_home = args.atb_part1
part2_home = args.atb_part2
part3_home = args.atb_part3

output_home = args.output_dir

if not os.path.exists(output_home):
    os.makedirs(output_home)

part1_ori_dir = path.join(part1_home, 'MOD', 'ATB1_v2_0', 'data', 'treebank', 'without-vowel')
part2_ori_dir = path.join(part2_home, 'data', 'treebank', 'without-vowel')
part3_ori_dir = path.join(part3_home, 'MOD', 'data', 'penntree', 'without-vowel')

train_split_file = path.join(args.split_dir, 'train.txt')
dev_split_file = path.join(args.split_dir, 'dev.txt')
test_split_file = path.join(args.split_dir, 'test.txt')


def get_splits(file):
    filenames = []
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            filenames.append(line)
    return filenames


train_files = get_splits(train_split_file)
dev_files = get_splits(dev_split_file)
test_files = get_splits(test_split_file)

split_dict = {'train': train_files, 'dev': dev_files, 'test': test_files}


def get_file_path(file_name):
    if file_name.startswith('2000'):
        return path.join(part1_ori_dir, file_name)
    if file_name.startswith('UMAAH'):
        return path.join(part2_ori_dir, file_name)
    if file_name.startswith('ANN'):
        return path.join(part3_ori_dir, file_name)


for flag in ['train', 'dev', 'test']:
    splits = split_dict[flag]
    with open(path.join(output_home, flag + '.raw.mrg'), 'w', encoding='utf8') as wf:
        print('processing %s ...' % flag)
        max_sent_len = 0
        ignore = 0
        sent_num = 0
        line_num = 0
        for filename in splits:
            file_path = get_file_path(filename)
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                left_num = 0
                splits = line.split()
                sub_line = []
                last_index = 0
                line_num += 1
                for index, item in enumerate(splits):
                    if index > 0 and left_num == 0:
                        sub_line.append(' '.join(splits[last_index: index]))
                        last_index = index
                        # line = '(S %s)' % line
                    left_num += item.count('(')
                    left_num -= item.count(')')
                last_sub_line = ' '.join(splits[last_index:])
                if left_num < 0:
                    last_sub_line = last_sub_line[:len(last_sub_line)+left_num]
                    # line = line[:len(line)+left_num]
                sub_line.append(last_sub_line)
                # sub_line.append(line)

                for sl in sub_line:
                    parse_tree = Tree.fromstring(sl)
                    sent_len = len(parse_tree.leaves())
                    # if sent_len > 40:
                    #     ignore += 1
                    #     continue
                    if sent_len > max_sent_len:
                        max_sent_len = sent_len
                    wf.write(' '.join(str(parse_tree).split()))
                    wf.write('\n')
                    sent_num += 1

        # print('sentence number in %s: %d' % (flag, sent_num))
        # print('line number in %s: %d' % (flag, line_num))
        # print('max sent length in %s: %d' % (flag, max_sent_len))
