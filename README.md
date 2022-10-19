# Crosslingual Complexity Transfer
All experiments can be run on a single GPU in under one hour. The code for multi-task learning was adapted from https://github.com/gsarti/interpreting-complexity.

# Structure

## `analysis`
This folder contains notebooks to analyse the data, evaluate the model predictions and plot the results. We use scikit-learn implementations for calculating R2 and explained variance. We use scipy for calculating Spearman correlations.

## `data`
Create a data folder with the following subfolders:
- `geco`

Folder for data from the Ghent Eye-tracking Corpus (Cop et al., 2017). The files `MonolingualReadingData` and `EnglishReadingMaterials` should be placed in this folder and can be downloaded here: https://expsy.ugent.be/downloads/geco/

- `meco`

Folder for data from the Multilingual Eye-tracking Corpus (Siegelman et al., 2022). The files `sentence_data.csv` and `supp texts.csv` should be placed in this folder and can be downloaded here: https://osf.io/3527a/

- `pud`

Folder for data from Parellel Universal Dependencies (Zeman et al., 2017). The CoNLL files from `UD_Korean-PUD`, `UD_English-PUD` and `UD_Turkish-PUD` should be placed in this folder and can be downloaded here: https://github.com/UniversalDependencies

## `preprocessing`

To preprocess the data, run the following scripts (which are located in the `preprocessing` folder):

- meco --> First, run `extract_meco_texts.py`. Then, calculate linguistic features for each txt file using the Profiling-UD tool (sentence-level analysis, presegmented text, http://linguistic-profiling.italianlp.it/). Download the csv file called "linguistic profile" and place it in its respective language folder (e.g. `meco/files_per_language/English`). Finally, run `preprocess_meco.py`.
- geco --> First, run `preprocess_geco.py`. Then, write the sentences from the 'text' column in the file `preprocessed_geco_sentence_level.tsv` to a txt file (1 sentence per line). Calculate linguistic features for the file using the Profiling-UD tool and place the resulting csv file in the `geco` folder. Finally, run `add_ling_feats_geco.py`.
- pud --> Write the sentences from each language-specific ConLL file to a txt file (1 sentence per line). Calculate linguistic features for each file using the Profiling-UD tool and place the resulting csv files in the `pud` folder. Then, run `preprocess_pud.py`.

## `scripts`

- `compute_baselines.py` --> for training and testing SVM models on different subsets of structural complexity features
- `finetune_sentence_level.py` --> for finetuning transformer models on sentence-level eye-tracking metrics using multi-task learning

# Running the code

Once all data has been preprocessed, it can be used to finetune XLM-R (or another transformer model). With the following command, the model is trained and evaluated on the English GECO data using 5-fold cross-validation and learns four eye-tracking features simultaneously (first-pass duration, fixation count, total fixation duration, regression duration):

```
python scripts/finetune_sentence_level.py --run_name train-xlm --data_dir data/eyetracking/geco/train_test --save_dir models/xlm-trained --model_name xlm-roberta-base --experiment_name eval-xlm-geco-5fold --pooling_strategy mean --label_columns scaled_first_pass_dur scaled_fix_count scaled_tot_fix_dur scaled_tot_regr_from_dur --folds 5 --num_train_epochs 15 --evaluate_every 40 --patience 5 --train_mode regression
```
To evaluate the model on any language from MECO, place the file `test.csv` corresponding to the language of interest in the `train_test` folder (e.g. `data/meco/files_per_language/Dutch/test.csv`). The parameter `--do_eval_only` loads the trained model and evaluates it on the MECO test data.
```
python scripts/finetune_sentence_level.py --run_name eval-meco-English --data_dir data/eyetracking/meco/files_per_language/English --model_name models/xlm-trained --experiment_name eval-meco-English --pooling_strategy mean --label_columns scaled_first_pass_dur scaled_fix_count scaled_tot_fix_dur scaled_tot_regr_from_dur --do_eval_only --folds 5 --train_mode regression
```
To probe linguistic knowledge in the model's representations, the same script can be used. In this case, the encoder model should be frozen using the parameter `--freeze_model`, so that only the final regression layer is fine-tuned. The following command probes the linguistic feature "lexical density" in the pre-trained representations of XLM-R, using the first fold of the English PUD data:
```
python scripts/finetune_sentence_level.py --freeze_model --data_dir data/pud/train_test_en/fold_0 --model_name xlm-roberta-base --label_columns scaled_lexical_density --run_name probe-lexical-density --train_mode regression --folds 0 --num_train_epochs 5 --evaluate_every 0 --freeze_model
```
To probe linguistic features in the representations of a trained model, use the path to the trained model as `--model_name`.
