# Cross-lingual Transfer of Cognitive Processing Complexity
Repository for the paper "Cross-lingual Transfer of Cognitive Processing Complexity". The work was carried out by Charlotte Pouw and supervised by dr. Lisa Beinborn. The code for multi-task learning was adapted from https://github.com/gsarti/interpreting-complexity.

# Structure

## `analyses`
This folder contains notebooks to analyse the data and to plot the results.

## `data`
- `geco`

Folder for data from the Ghent Eye-tracking Corpus (Cop et al., 2017). The files `MonolingualReadingData` and `EnglishReadingMaterials` should be placed in this folder and can be downloaded here: https://expsy.ugent.be/downloads/geco/

- `meco`

Folder for data from the Multilingual Eye-tracking Corpus (Siegelman et al., 2022). The files `sentence_data.csv` and `supp texts.csv` should be placed in this folder and can be downloaded here: https://osf.io/3527a/

- `pud`

Folder for data from Parellel Universal Dependencies (Zeman et al., 2017). The CoNLL files from `UD_Korean-PUD`, `UD_English-PUD` and `UD_Turkish-PUD` should be placed in this folder and can be downloaded here: https://github.com/UniversalDependencies

## `preprocessing`

To preprocess the data, run the following scripts (which are located in the `preprocessing` folder):

- meco --> first, run `extract_meco_text.py` and subsequently run `preprocess_meco.py`
- geco --> run `preprocess_geco.py`
- pud --> run `preprocess_pud.py`

# Running the code

Once all data has been preprocessed, it can be used to finetune XLM-RoBERTa (or another transformer model). With the following command, the model is trained and evaluated on the English GECO data and learns four eye-tracking features simultaneously (first-pass duration, fixation count, total fixation duration, regression duration):

```
python finetune_xlm.py --data_dir ./data/geco/train_test --label_columns scaled_first_pass_dur scaled_fix_count scaled_tot_fix_dur scaled_tot_regr_from_dur --run_name train-xlm-on-geco
```
To evaluate the model on any language from MECO, place the file `test.csv` corresponding to the language of interest in the `train_test` folder (e.g. `data/meco/files_per_language/Dutch/test.csv`).

To probe the linguistic knowledge that is encoded in the model's representations, the same script can be used. In this case, the encoder model should be frozen, so that only the final regression layer is fine-tuned. The following command probes the linguistic feature "lexical density", using the first fold of the English PUD data:
```
python finetune_sentence_level.py --freeze_model --data_dir ./data/pud/train_test_en/fold_0 --label_columns scaled_lexical_density --run_name probe-lexical-density
```
