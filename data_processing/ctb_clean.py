from os import path
import argparse

dev_fix = {
    '(TOP (S 他/PN 说/VV ，/PU 近年/NT 来/LC ，/PU 中国/NR 经济/NN 发展/NN 迅速/VA ，/PU 泰国/NR 政府/NN 和/CC 人民/NN 感到/VV *PRO*/-NONE- 十分/AD 高兴/VA 。/PU))': '(TOP (IP (NP (PN 他)) (VP (VV 说) (PU ，) (IP (IP (LCP (NT 近年) (LC 来)) (PU ，) (NP (NP (NR 中国)) (NP (NN 经济))) (NP (NN 发展)) (VP (VA 迅速))) (PU ，) (IP (NP (NP (NR 泰国)) (NP (NN 政府) (CC 和) (NN 人民))) (VP (VV 感到) (IP (VP (ADVP (AD 十分)) (VP (VA 高兴)))))))) (PU 。)))',
    '(TOP (S 专访/NN ：/PU 国家/NN 女足/NN 教练/NN 谈/VV 近期/NT “/PU 三/CD 步/M 曲/NN ”/PU))': '(TOP (NP (NP (NN 专访)) (PU ：) (IP (NP (NN 国家) (NN 女足) (NN 教练)) (VP (VV 谈) (NP (NP (NT 近期)) (NP (PU “) (QP (CD 三) (CLP (M 步))) (NP (NN 曲)) (PU ”)))))))'
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True, type=str)

args = parser.parse_args()

print('Removing bad sentences ...')

for flag in ['train', 'dev', 'test']:
    input_file = path.join(args.input_dir, flag + '.raw.mrg')
    output_file = path.join(args.output_dir, flag + '.mrg')
    with open(input_file, 'r', encoding='utf8') as fr, open(output_file, 'w', encoding='utf8') as fw:
        lines = fr.readlines()
        removed = 0
        for line in lines:
            line = line.strip()
            if line.count(')') == 2:
                if flag == 'train':
                    removed += 1
                if flag == 'dev':
                    fix_line = dev_fix[line]
                    fw.write(fix_line)
                    fw.write('\n')
                continue
            fw.write(line)
            fw.write('\n')
        print('%s\t%d' % (flag, removed))
