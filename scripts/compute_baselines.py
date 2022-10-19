from tqdm import tqdm
from sklearn.svm import SVR
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

import argparse
import pandas as pd
import numpy as np
import os

from utils import get_avg_word_length_with_punct, get_avg_token_freq, get_n_low_freq_words, scale

def svm(df_train, df_test, target_label, setting, n_fold):

    if setting == 'linguistic':
        target_features = ['lexical_density',                                                            # morpho-syntactic
                           'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent']   # syntactic

    elif setting == 'length':
        target_features = ['token_count', 'avg_word_length']

    elif setting == 'frequency':
        target_features = ['avg_token_freq', 'n_low_freq_words']

    elif setting == 'all':
        target_features = ['token_count', 'avg_word_length',                                         # surface
                           'lexical_density',                                                        # morpho-syntactic
                           'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent', # syntactic
                           'avg_token_freq', 'n_low_freq_words']                                     # frequency

    X_train = df_train[target_features]
    y_train = scale(df_train[target_label].tolist())
    X_test = df_test[target_features]
    y_test = scale(df_test[target_label])

    # Get predictions from SVR
    np.random.seed(42)
    svr = SVR(kernel="linear")
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    preds_path = f'results/preds-{setting}-svm-{target_label}-English-{n_fold}.tsv'

    # Write out predictions + true values
    with open(preds_path, 'w', encoding='utf8') as preds_file:
        for pred, label in zip(y_pred, y_test):
            preds_file.write(target_label + '\t' + str(pred) + '\t' + str(label) + '\n')


def svm_kfold(args):

    # Load GECO train data
    df_train = pd.read_csv(args.geco_path, sep='\t')
    df_train['avg_token_freq'] = get_avg_token_freq(df_train['text'].tolist(), 'English')
    df_train['n_low_freq_words'] = get_n_low_freq_words(df_train['text'].tolist(), 'English')
    df_train['avg_word_length'] = get_avg_word_length_with_punct(df_train['text'].tolist())

    # Load MECO test data
    df_test = pd.read_csv(f'{args.meco_path}/English/test.tsv', sep='\t')

    # Perform cross-validation using the MECO test set each time (i.e. we only vary the training data)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0
    for train_idx, test_idx in tqdm(kf.split(df_train)):
        train = df_train.iloc[train_idx]
        test = df_test

        # Iterate over target labels
        for target_label in ['scaled_fix_count', 'scaled_first_pass_dur', 'scaled_tot_fix_dur', 'scaled_tot_regr_from_dur']:

            # Calculate four different SVM baselines for English
            for setting in ['linguistic', 'length', 'frequency', 'all']:
                svm(train, test, target_label, setting, fold)
        fold += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geco_path",
        type=str,
        default="data/geco/data_with_ling_feats.tsv",
        help="Path to the GECO data with concatenated linguistic features",
    )
    parser.add_argument(
        "--meco_path",
        type=str,
        default="data/meco/files_per_language",
        help="Path to the MECO data per language",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/eye-movement-prediction/baseline_results_new.csv",
        help="Path to the file where the baseline results will be written",
    )

    args = parser.parse_args()
    svm_kfold(args)