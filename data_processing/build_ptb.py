from os.path import join
import os
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import glob

train_splits = ["0" + str(i) for i in range(2, 10)] + [str(i) for i in range(10, 22)]
test_splits = ["23"]
dev_splits = ["22"]


def glob_files(data_root, splits):
    return [fname for split in splits for fname in sorted(glob.glob(join(data_root, split, "*.mrg")))]


def write_to_file(data_root, splits, outfile, add_top=False):
    reader = BracketParseCorpusReader('.', glob_files(data_root, splits))
    with open(outfile, 'w') as f:
        for tree in reader.parsed_sents():
            tree_rep = tree.pformat(margin=1e100)
            if add_top:
                tree_rep = "(TOP %s)" % tree_rep
            assert('\n' not in tree_rep)
            f.write(tree_rep)
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_home", required=True)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--add_top", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_dir = join(args.data_home, 'RAW/parsed/mrg/wsj')

    write_to_file(input_dir, train_splits, join(args.output_dir, 'train.ori.mrg'), args.add_top)
    write_to_file(input_dir, dev_splits, join(args.output_dir, 'dev.ori.mrg'), args.add_top)
    write_to_file(input_dir, test_splits, join(args.output_dir, 'test.ori.mrg'), args.add_top)
