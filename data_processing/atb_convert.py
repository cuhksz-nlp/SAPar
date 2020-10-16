from os.path import join
from nltk.tree import Tree

BUCKWALTER_MAP = {
    '\'': '\u0621',
    '|': '\u0622',
    '>': '\u0623',
    'O': '\u0623',
    '&': '\u0624',
    'W': '\u0624',
    '<': '\u0625',
    'I': '\u0625',
    '}': '\u0626',
    'A': '\u0627',
    'b': '\u0628',
    'p': '\u0629',
    't': '\u062A',
    'v': '\u062B',
    'j': '\u062C',
    'H': '\u062D',
    'x': '\u062E',
    'd': '\u062F',
    '*': '\u0630',
    'r': '\u0631',
    'z': '\u0632',
    's': '\u0633',
    '$': '\u0634',
    'S': '\u0635',
    'D': '\u0636',
    'T': '\u0637',
    'Z': '\u0638',
    'E': '\u0639',
    'g': '\u063A',
    '_': '\u0640',
    'f': '\u0641',
    'q': '\u0642',
    'k': '\u0643',
    'l': '\u0644',
    'm': '\u0645',
    'n': '\u0646',
    'h': '\u0647',
    'w': '\u0648',
    'Y': '\u0649',
    'y': '\u064A',
    'F': '\u064B',
    'N': '\u064C',
    'K': '\u064D',
    'a': '\u064E',
    'u': '\u064F',
    'i': '\u0650',
    '~': '\u0651',
    'o': '\u0652',
    '`': '\u0670',
    '{': '\u0671',
}

BUCKWALTER_UNESCAPE = {
    # "-LRB-": "(",
    # "-RRB-": ")",
    # "-LCB-": "{",
    # "-RCB-": "}",
    # "-LSB-": "[",
    # "-RSB-": "]",
    # '-PLUS-': "+",
    # '-MINUS-': "-",
}

BUCKWALTER_UNCHANGED = set('.?!,"%-/:;=')

HEBREW_MAP = {
    'A': '\u05d0',
    'B': '\u05d1',
    'G': '\u05d2',
    'D': '\u05d3',
    'H': '\u05d4',
    'W': '\u05d5',
    'Z': '\u05d6',
    'X': '\u05d7',
    'J': '\u05d8',
    'I': '\u05d9',
    'K': '\u05db',
    'L': '\u05dc',
    'M': '\u05de',
    'N': '\u05e0',
    'S': '\u05e1',
    'E': '\u05e2',
    'P': '\u05e4',
    'C': '\u05e6',
    'Q': '\u05e7',
    'R': '\u05e8',
    'F': '\u05e9',
    'T': '\u05ea',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    'U': '"',
    'O': '%',
    '.': '.',
    ',': ',',
}

HEBREW_SUFFIX_MAP = {
    '\u05db': '\u05da',
    '\u05de': '\u05dd',
    '\u05e0': '\u05df',
    '\u05e4': '\u05e3',
    '\u05e6': '\u05e5',
}

HEBREW_UNESCAPE = {
    "yyCLN": ":",
    "yyCM": ",",
    "yyDASH": "-",
    "yyDOT": ".",
    "yyELPS": "...",
    "yyEXCL": "!",
    "yyLRB": "(",
    "yyQM": "?",
    "yyRRB": ")",
    "yySCLN": ";",
}


def arabic(inp):
    """
    Undo Buckwalter transliteration

    See: http://languagelog.ldc.upenn.edu/myl/ldc/morph/buckwalter.html

    This code inspired by:
    https://github.com/dlwh/epic/blob/master/src/main/scala/epic/util/ArabicNormalization.scala
    """
    return "".join(
        BUCKWALTER_MAP.get(char, char)
        for char in BUCKWALTER_UNESCAPE.get(inp, inp))

import argparse

# MOD/ATB1_v2_0/data/treebank/with-vowel

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True)

args = parser.parse_args()

for flag in ['train', 'dev', 'test']:
    input_file = join(args.input_dir, flag + '.cleaned.mrg')
    output_file = join(args.output_dir, flag + '.mrg')

    data = []
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.count(')') == 2:
                continue
            parse_tree = Tree.fromstring(line)
            leaves = parse_tree.leaves()
            words = []
            for l in leaves:
                # l = l[::-1]
                words.append(arabic(l))
            # print(' '.join(words[::1]))
            for i, s in enumerate(parse_tree.subtrees(lambda t: t.height() == 2)):
                s[0] = words[i]
            data.append(' '.join(str(parse_tree).split()))

    with open(output_file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(d + '\n')
