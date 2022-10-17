from utils import scale
import argparse
import logging
import os
import string

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('C:/Users/charl/Documents/GitHub/charlotte-pouw-crosslingual-transfer-of-linguistic-complexity/src/')

from lingcomp.data_utils import GECOProcessor
from lingcomp.script_utils import save_tsv, train_test_split_sentences

def preprocess_geco_data(args):
    dfs = []
    processor = GECOProcessor(
        args.data_dir,
        fillna=args.fillna_strategy,
    )
    df = processor.get_sentence_data(
        args.eyetracking_participant,
        min_len=args.eyetracking_min_sent_len,
        max_len=args.eyetracking_max_sent_len,
        )
    # Common format across sentence-level datasets used in FARM
    df.rename(columns={"sentence": "text"}, inplace=True)
    dfs.append(df)
    if len(dfs) > 1:
        df = pd.concat(dfs, ignore_index=True)
        df = reindex_sentence_df(df)

    # add scaled features
    for feature in ['first_pass_dur', 'fix_count', 'tot_fix_dur', 'tot_regr_from_dur']:
        df[f'scaled_{feature}'] = scale(df[feature].tolist())

    out = os.path.join(args.out_dir, f"preprocessed_geco_sentence_level.tsv")
    save_tsv(df, out)
    logging.info(f"Eyetracking data were preprocessed and saved as" f" {out} with shape {df.shape}")
    return df


def do_train_test_split(args):
    logging.info("Performing train-test split...")
    folder = f"{args.data_dir}/train_test"
    if not os.path.exists(folder):
        os.makedirs(folder)
        train, test = train_test_split(args.et, test_size=args.test_size, random_state=args.seed)
        save_tsv(train, f"{folder}/train.tsv")
        save_tsv(test, f"{folder}/test.tsv")
        logging.info(f"Train-test data saved in {folder}")
    else:
        logging.info("Train-test data already exist in path, not overriding them.")

def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eyetracking", action="store_true", default=True)
    parser.add_argument("--data_dir", type=str, default="./data/geco")
    parser.add_argument("--out_dir", type=str, default="./data/geco/preprocessed")
    parser.add_argument("--do_train_test_split", default=True, action="store_true")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eyetracking_participant",
        type=str,
        default="avg",
        help="Participant selected for eyetracking scores. Default is avg of all participants.",
    )
    parser.add_argument("--eyetracking_min_sent_len", type=int, default=5)
    parser.add_argument("--eyetracking_max_sent_len", type=int, default=45)
    parser.add_argument(
        "--fillna_strategy",
        default="zero",
        type=str,
        help="Specifies the NaN filling strategy for eyetracking processors.",
        choices=["none", "zero", "min_participant", "mean_participant", "max_participant"],
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.eyetracking:
        args.et = preprocess_geco_data(args)
    if args.do_train_test_split:
        do_train_test_split(args)

if __name__ == "__main__":
    preprocess()
