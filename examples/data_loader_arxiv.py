import os
import random
from argparse import ArgumentParser
from glob import glob

from nltk import word_tokenize

random.seed(42)


def format(fulltext_dir, summary_dir, output_dir, dev_split):
    full_texts = glob(os.path.join(fulltext_dir, "*.text"))
    full_texts.sort()
    summaries = glob(os.path.join(summary_dir, "*.summary"))
    summaries.sort()

    data = []

    for f, s in zip(full_texts, summaries):
        with open(f) as fr:
            f_c = ' '.join(
                word_tokenize(fr.read().lower().strip().replace("\n", " ")))
        with open(s) as fr:
            f_s = ' '.join(
                word_tokenize(fr.read().lower().strip().replace("\n", " ")))
        data.append((f_c, f_s))

    random.shuffle(data)
    max_length = len(data)

    split = int((1 - dev_split) * max_length)

    train = data[0:split]
    test = data[split:]

    with open(os.path.join(output_dir, 'train.tsv'), 'w') as fout:
        for obj in train:
            fout.write('{}\t{}\n'.format(obj[0], obj[1]))
    with open(os.path.join(output_dir, 'test.tsv'), 'w') as fout:
        for obj in test:
            fout.write('{}\t{}\n'.format(obj[0], obj[1]))


def main():
    parser = ArgumentParser("Format dataset for IBM PyTorch Seq2Seq")
    parser.add_argument("-f", "--fulltext_dir", help="Full text directory")
    parser.add_argument("-s", "--summary_dir", help="Summary directory")
    parser.add_argument("-o", "--output_dir", help="Output directory")
    parser.add_argument(
        "-d", "--dev_split", type=float, help="Dev split", default=0.8)
    args = parser.parse_args()

    format(args.fulltext_dir, args.summary_dir, args.output_dir,
           args.dev_split)


if __name__ == '__main__':
    main()
