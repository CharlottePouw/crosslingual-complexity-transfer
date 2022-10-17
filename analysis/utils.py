import string, os
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency
import random

from sklearn.preprocessing import MinMaxScaler

# Map to look up languages in the wordfreq package
language_map = {'Dutch': 'nl',
                'English': 'en',
                'Finnish': 'fi',
                'German': 'de',
                'Greek': 'el',
                'Hebrew': 'he',
                'Italian': 'it',
                'Korean': 'ko',
                'Norwegian': 'nb',
                'Russian': 'ru',
                'Spanish': 'es',
                'Turkish': 'tr'}

def scramble_tokens(sentences):

    scrambled_sents = []
    np.random.seed(42)

    for sentence in sentences:
        # split sentence based on spaces
        tokens = sentence.split(' ')
        # shuffle the order of the tokens
        random.shuffle(tokens)
        # save scrambled sentence
        scrambled_sents.append(' '.join(tokens))

    return scrambled_sents

def get_avg_token_freq(sentences, language):
    '''
    For each sentence, compute average token frequency (Zipf), excluding punctuation.
    '''
    avg_token_freqs = []

    for sentence in sentences:

        # remove punctuation
        clean_sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        # split sentence based on spaces
        tokens = clean_sentence.split(' ')

        # get zipf frequency of each token
        freqs = []
        for token in tokens:
            freq = zipf_frequency(token, language_map[language], wordlist='best', minimum=1.0)
            freqs.append(freq)

        # calculate average token frequency in the sentence
        avg_token_freqs.append(np.mean(freqs))

    return avg_token_freqs


def get_n_low_freq_words(sentences, language, threshold=4):
    '''
    For each sentence, count the number of words with a Zipf frequency below 4.
    '''
    num_low_freq_words_per_sent = []

    for sentence in sentences:

        num_low_freq_words = 0

        # remove punctuation
        clean_sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        # split sentence based on spaces
        tokens = clean_sentence.split(' ')

        # get zipf frequency of each token
        for token in tokens:
            freq = zipf_frequency(token, language_map[language], wordlist='best', minimum=1.0)

            # check if the frequency is below the treshold --> if so, count it as low-frequency
            if freq < threshold:
                num_low_freq_words += 1
            else:
                # if the word is not in the dictionary, continue
                continue

        num_low_freq_words_per_sent.append(num_low_freq_words)

    return num_low_freq_words_per_sent


def get_avg_word_length_with_punct(sentences):
    '''
    For each sentence, compute average word length with attached punctuation included.
    '''
    avg_word_lengths = []

    for sentence in sentences:
        # split sentence based on spaces
        words = sentence.split(' ')

        # get word lengths
        word_lengths = [len(word) for word in words]

        # calculate average word length in the sentence
        avg_word_lengths.append(np.mean(word_lengths))

    return avg_word_lengths

def scale(feature_values):

    scaler = MinMaxScaler(feature_range=(0,100))

    # reshape 1D array to 2D
    reshaped_feature_list = np.array(feature_values).reshape(-1, 1)

    # scale feature values between 0 and 100
    scaled_feature_values = scaler.fit_transform(reshaped_feature_list)

    # flatten the 2D array to 1D
    flat_scaled_feature_values = np.ravel(scaled_feature_values)

    return flat_scaled_feature_values


def get_meco_df(language):
    path = f'./data/meco/files_per_language/{language}/test.tsv'

    if os.path.exists(path):
        df = pd.read_csv(f'./data/meco/files_per_language/{language}/test.tsv', sep='\t', encoding='utf8')
        df['dataset'] = len(df) * [f'{language}']

    return df