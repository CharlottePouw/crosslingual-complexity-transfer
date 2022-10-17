import os
import numpy as np

meco_path = 'data/meco/files_per_language'

for language_folder in os.listdir(meco_path):
    with open(f'{meco_path}/{language_folder}/{language_folder.lower()}_clean.txt', encoding='utf8') as infile:

        # Retrieve the sentences
        sentences = infile.readlines()
        sentences = [s.strip('\n') for s in sentences]
        splitted_sents = [sent.split(' ') for sent in sentences]

        sent_lengths = [len(sent) for sent in splitted_sents]
        word_lengths = [len(word) for sent in splitted_sents for word in sent]
        avg_sent_length = np.mean(sent_lengths)
        avg_word_length = np.mean(word_lengths)

        print('Statistics for', language_folder, ':')
        print('Num sents:', len(sentences))
        print('Num words:', len(word_lengths))
        print('Avg sent length:', avg_sent_length)
        print('Avg word length', avg_word_length)
        print()