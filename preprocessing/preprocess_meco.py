import pandas as pd
import os, csv, random
import numpy as np

from utils import get_avg_word_length_with_punct, get_avg_token_freq, get_n_low_freq_words, scale, scramble_tokens

meco_path = 'data/meco/files_per_language'
eyetracking_path = 'data/meco/sentence_data.csv'

# Read in the eyetracking data
et_data = pd.read_csv(eyetracking_path, encoding='utf8', quotechar='"')

# Calculate regression path duration: go-past time minus selective go-past time
regr_durs = et_data['firstrun.gopast'] - et_data['firstrun.gopast.sel']
et_data['tot_regr_from_dur'] = regr_durs

# Fix an alignment problem with certain participants
unaligned_participants = ['ee_22', 'ee_9', 'ru_8']

for p in unaligned_participants:
    et_data = et_data[et_data['uniform_id'] != p]

#Average over participants: Get the average of the eyetracking features per sentence
avg_et_data = et_data.groupby(['lang', 'trialid', 'sentnum'], as_index=False).mean()

# Map the ET feature names from MECO to GECO standards
avg_et_data.rename(columns={'firstrun.dur':'first_pass_dur','nfix':'fix_count','dur':'tot_fix_dur'},inplace=True)
#
# for name, group in avg_et_data.groupby('lang'):
#     print(name, len(group))

# Define features that we want to scale
target_features = ['first_pass_dur', 'fix_count', 'tot_fix_dur', 'tot_regr_from_dur',          # eye-tracking
                   'token_count', 'avg_word_length',                                           # surface
                   'lexical_density',                                                          # morpho-syntactic
                   'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent',  # syntactic
                   'avg_token_freq', 'n_low_freq_words']                                       # frequency

#for language_folder in os.listdir(meco_path):
for language_folder in ['English']:
    if language_folder != 'English':
        with open(f'{meco_path}/{language_folder}/{language_folder.lower()}_clean.txt', encoding='utf8') as infile:

            # Retrieve the sentences
            sentences = infile.readlines()
            sentences = [s.strip('\n') for s in sentences]

            # Get the key for the language in the eyetracking dataframe (e.g. 'du' for Dutch)
            if language_folder == 'Turkish':
                language_key = 'tr'
            elif language_folder == 'Estonian':
                language_key = 'ee'
            else:
                language_key = language_folder.lower()[:2]

            # Get the language-specific dataframe with eyetracking features
            et_feats = avg_et_data.groupby('lang', as_index=False).get_group(language_key)
            et_feats.reset_index(drop=True, inplace=True)

            permutation = True
            scrambled = False
            # Add a column with the sentences to the language-specific dataframe
            # Scramble tokens if specified
            if scrambled:
                et_feats.insert(3, 'text', scramble_tokens(sentences))
            # Shuffle the input/output pairs if specified
            elif permutation:
                np.random.seed(42)
                random.shuffle(sentences)
                et_feats.insert(3, 'text', sentences)
            else:
                et_feats.insert(3, 'text', sentences)

            # Read in linguistic features and add them to the dataframe
            ling_feats = pd.read_csv(f'{meco_path}/{language_folder}/{language_folder.lower()}_feats.csv', sep='\t', header=0, encoding='utf8', quoting = csv.QUOTE_NONE)
            ling_feats.reset_index(drop=True, inplace=True)

            # Concatenate the two dataframes
            final_data = pd.concat([et_feats, ling_feats], axis=1)
#
            # add frequency and length columns to the dataframe (exclude freq feats for korean and estonian)
            if language_folder != 'Korean' and language_folder != 'Estonian':
                final_data['avg_token_freq'] = get_avg_token_freq(final_data['text'].tolist(), language_folder)
                final_data['n_low_freq_words'] = get_n_low_freq_words(final_data['text'].tolist(), language_folder)

            final_data['avg_word_length'] = get_avg_word_length_with_punct(final_data['text'].tolist())
            final_data = final_data.rename(columns={"sent.nwords": "token_count"})

            # Add columns with scaled features to the frame (exclude freq feats for korean and estonian)
            if language_folder == 'Korean' or language_folder == 'Estonian':
                target_features = ['first_pass_dur', 'fix_count', 'tot_fix_dur', 'tot_regr_from_dur',           # eye-tracking
                                   'token_count', 'avg_word_length',                                            # surface
                                   'lexical_density',                                                           # morpho-syntactic
                                   'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent']   # syntactic
                                   #'avg_token_freq', 'n_low_freq_words']  # frequency

            for feature in target_features:
                final_data[f'scaled_{feature}'] = scale(final_data[feature].tolist())

        if scrambled:
            print(f"Writing scrambled data for {language_folder} to tsv")
            final_data.to_csv(f'{meco_path}_scrambled/{language_folder}/test.tsv', encoding='utf8', sep='\t', index=False, quotechar='"')
        elif permutation:
            print(f"Writing permuted data for {language_folder} to tsv")
            final_data.to_csv(f'{meco_path}_permuted/{language_folder}/test.tsv', encoding='utf8', sep='\t', index=False, quotechar='"')
        else:
            print(f"Writing data for {language_folder} to tsv")
            final_data.to_csv(f'{meco_path}/{language_folder}/test.tsv', encoding='utf8', sep='\t', index=False, quotechar='"')
    else:
        english_df = pd.read_csv(f'{meco_path}/english/test.tsv', sep='\t')
        sentences = english_df['text'].tolist()

        permutation = True
        scrambled = False
        # Add a column with the sentences to the language-specific dataframe
        # Scramble tokens if specified
        if scrambled:
            english_df['text'] = scramble_tokens(sentences)
        # Shuffle the input/output pairs if specified
        elif permutation:
            np.random.seed(42)
            random.shuffle(sentences)
            english_df['text'] = sentences

        if scrambled:
            print(f"Writing scrambled data for {language_folder} to tsv")
            english_df.to_csv(f'{meco_path}_scrambled/{language_folder}/test.tsv', encoding='utf8', sep='\t', index=False, quotechar='"')
        elif permutation:
            print(f"Writing permuted data for {language_folder} to tsv")
            english_df.to_csv(f'{meco_path}_permuted/{language_folder}/test.tsv', encoding='utf8', sep='\t', index=False, quotechar='"')


