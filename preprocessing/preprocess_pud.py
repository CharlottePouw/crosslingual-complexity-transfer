import pandas as pd
import csv
import argparse
import random
import os

from utils import get_avg_word_length_with_punct, get_avg_token_freq, get_n_low_freq_words, scale

def preprocess_pud(conll_path, feats_path, out_dir, language):

    with open(conll_path, 'r', encoding='utf8') as infile:
        lines = infile.readlines()
        sentences = [line.lstrip('# text = ').rstrip('\n') for line in lines if line.startswith('# text = ')]

    df = pd.read_csv(feats_path, sep='\t', header=0, encoding='utf8', quoting=csv.QUOTE_NONE)

    # Add custom frequency and length columns to the train dataframe
    df['text'] = sentences
    df['token_count'] = [len(sent.split(' ')) for sent in sentences]
    df['avg_token_freq'] = get_avg_token_freq(df['text'].tolist(), language)
    df['n_low_freq_words'] = get_n_low_freq_words(df['text'].tolist(), language)
    df['avg_word_length'] = get_avg_word_length_with_punct(df['text'].tolist())

    # Define features that we want to scale
    target_features = ['token_count', 'avg_word_length',  # surface
                       'lexical_density',  # morpho-syntactic
                       'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent',  # syntactic
                       'avg_token_freq', 'n_low_freq_words']  # frequency

    # Add columns with scaled features to the frame
    for feature in target_features:
        df[f'scaled_{feature}'] = scale(df[feature].tolist())

    out_path = f'{out_dir}/preprocessed_pud_{language}.tsv'

    df.to_csv(out_path, sep='\t', encoding='utf8', quoting=csv.QUOTE_NONE, index=False)

    return out_path

def train_test_split(out_path, train_test_dir, num_folds=5):

    with open(out_path, 'r', encoding='utf8') as f:
        data = f.read().split('\n')

    header = data[0]

    # skip the header and the last element, which is an empty row
    new_data = data[1:-1]
    random.shuffle(new_data)

    batches = {'batch_0': new_data[:200],
               'batch_1': new_data[200:400],
               'batch_2': new_data[400:600],
               'batch_3': new_data[600:800],
               'batch_4': new_data[800:1000]}

    for k in range(num_folds):

        print(len(batches[f'batch_{k}']))

        directory = f'{train_test_dir}/fold_{k}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        train_data = []
        for batch_name, batch in batches.items():
            if batch_name != f'batch_{k}':
                train_data += batch

        test_data = batches[f'batch_{k}']

        train_path = directory+'/train.tsv'
        test_path = directory+'/test.tsv'

        with open(train_path, 'w', encoding='utf8') as outfile:
            outfile.write(header + '\n')
            for line in train_data:
                outfile.write(line + '\n')

        with open(test_path, 'w', encoding='utf8') as outfile:
            outfile.write(header + '\n')
            for line in test_data:
                outfile.write(line + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--conll_path',
                        type=str,
                        help='path to conll file that contains sentences',
                        default='data/pud/en_pud-ud-test.conllu')
    parser.add_argument('--feats_path',
                        type=str,
                        help='path to csv file that contains linguistic profiling features',
                        default='data/pud/ling_feats_pud_english.tsv')
    parser.add_argument('--out_dir',
                        type=str,
                        help='directory where the preprocessed data is stored',
                        default='data/pud')
    parser.add_argument('--language',
                        type=str,
                        help='language used in the file',
                        default='english')
    parser.add_argument('--train_test_dir',
                        type=str,
                        help='directory where the train/test splits are stored',
                        default='data/pud/train_test_english')
    args = parser.parse_args()

    data_path = preprocess_pud(args.conll_path, args.feats_path, args.out_dir, args.language)
    train_test_split(data_path, args.train_test_dir)




