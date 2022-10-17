import pandas as pd
import os

n_sents_map = {'Dutch': 112,
    	       'Estonian': 112,
               'English': 99,
               'Finnish': 110,
               'German': 115,
               'Greek': 99, # is parsed as 101, fixed manually
               'Hebrew': 121,
               'Italian': 90,
               'Korean': 101, # is parsed as 102, fixed manually (meco problem)
               'Norwegian': 116, # is parsed as 117, fixed manually (meco problem)
               'Russian': 101,
               'Spanish': 98, # is parsed as 99, fixed manually (meco problem)
               'Turkish': 104}

data = pd.read_csv("data/meco/supp texts.csv")

# Rename the first column
data.rename(columns={'Unnamed: 0':'Language'}, inplace=True )

# Remove the last two columns (they are empty)
data = data.iloc[: , :-1]
data = data.iloc[: , :-1]

# Remove empty rows
data.dropna(subset = ["Language"], inplace=True)

# Transpose dataframe, such that each column is a language
data = data.set_index("Language")
data = data.T

# Get the list of languages
languages = list(data.columns)

#for language in languages:
for language in languages:
    # Create a folder for each language
    output_dir = f'data/meco/files_per_language/{language}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all texts for each language
    texts = data[language].tolist()

    # Add newlines to the texts, such that each sentence is written on a new line
    texts_with_newlines = [text.replace('.', '.\n') for text in texts]

    # Write the texts to separate txt files
    with open(f"{output_dir}/{language.lower()}.txt", 'w', encoding='utf8') as outfile:
        for text in texts_with_newlines:
            outfile.write(text)

    # Clean up files
    with open(f"{output_dir}/{language.lower()}.txt", 'r', encoding='utf8') as infile:
        lines = infile.readlines()

        # Remove empty lines
        lines = [line for line in lines if len(line) > 1]

        with open(f"{output_dir}/{language.lower()}_clean.txt", 'w', encoding='utf8') as outfile:
            for i, line in enumerate(lines):
                line = line.strip('"')
                line = line.strip(' ')
                line = line.replace('\\n', '')
                line = line.strip('\n')

                # If it is the final line, do not add a newline character at the end
                if i == (len(lines) - 1):
                    outfile.write(line)
                else:
                    outfile.write(line + '\n')
        
        if len(lines) != n_sents_map[language]:
            print(f'Number of {language} sents: {len(lines)} should be {n_sents_map[language]}. Fix this manually')