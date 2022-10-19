import pandas as pd
import numpy as np
import csv

from utils import get_avg_word_length_with_punct, get_avg_token_freq, get_n_low_freq_words, scale

# Load GECO sentences and eye-tracking data
geco_train_english = pd.read_csv('data/geco/preprocessed/preprocessed_geco_sentence_level.tsv', sep='\t', encoding='utf8', quoting=csv.QUOTE_NONE)
geco_train_english = geco_train_english[['participant', 'text_id', 'sentence_id', 'text', 'token_count',
                                         'fix_count', 'fix_prob', 'mean_fix_dur', 'first_fix_dur', 'first_pass_dur',
                                         'tot_fix_dur', 'refix_count', 'reread_prob', 'tot_regr_from_dur']]

# Load corresponding linguistic features
en_geco_train_feats = pd.read_csv('data/geco/ling_feats.csv', sep='\t')
en_geco_train_feats = en_geco_train_feats.drop('Filename', axis=1)
en_geco_train_feats.reset_index(drop=True)

# Concatenate the two dataframes
en_train_cols = geco_train_english.columns.to_list() + en_geco_train_feats.columns.to_list()
en_train_concatenation = np.concatenate([geco_train_english, en_geco_train_feats], axis=1)
en_geco_train_full = pd.DataFrame(en_train_concatenation, columns=en_train_cols)

# Add custom frequency and length columns to the train dataframe
en_geco_train_full['avg_token_freq'] = get_avg_token_freq(en_geco_train_full['text'].tolist(), 'English')
en_geco_train_full['n_low_freq_words'] = get_n_low_freq_words(en_geco_train_full['text'].tolist(), 'English')
en_geco_train_full['avg_word_length'] = get_avg_word_length_with_punct(en_geco_train_full['text'].tolist())

# Define features that we want to scale
target_features = ['first_pass_dur', 'fix_count', 'tot_fix_dur', 'tot_regr_from_dur',          # eye-tracking
                   'token_count', 'avg_word_length',                                           # surface
                   'lexical_density',                                                          # morpho-syntactic
                   'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent',  # syntactic
                   'avg_token_freq', 'n_low_freq_words']                                       # frequency

# Add columns with scaled features to the frame
for feature in target_features:
    en_geco_train_full[f'scaled_{feature}'] = scale(en_geco_train_full[feature].tolist())

en_geco_train_full.to_csv('data/geco/data_with_ling_feats.tsv', encoding='utf8', sep='\t', quoting=csv.QUOTE_NONE, index=False)