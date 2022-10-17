import csv
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import math

from lingcomp.data_utils.const import (
    FILLNA_COLUMNS,
    GECO_DATA_COLS,
    GECO_MATERIAL_COLS,
    GECO_NA_VALUES,
    GECO_POS_MAP,
    OUT_TYPES_SENTENCE,
    OUT_TYPES_WORD,
    TEXT_COLUMNS,
)
from lingcomp.script_utils import apply_parallel, read_tsv, save_tsv


logger = logging.getLogger(__name__)


class EyetrackingProcessor:
    """ Abstraction for a reader that converts eye-tracking data in a preprocessed format """

    def __init__(
        self,
        data_dir,
        data_filename,
        out_filename,
        ref_participant,
        text_columns=TEXT_COLUMNS,
        out_types_word=OUT_TYPES_WORD,
        out_types_sentence=OUT_TYPES_SENTENCE,
        nan_cols=FILLNA_COLUMNS,
        fillna="zero",
        **kwargs,
    ):
        """
        data_dir: Directory where eye-tracking dataset and materials are contained.
        data_filename: Name of main file in the data_dir containing eye-tracking measurements.
        out_filename: File where the preprocessed output will be saved.
        text_columns: Names of columns to be treated as text during aggregation
        ref_participant: The name of the reference participant having annotated all examples,
            used for grouping and averaging scores.
        out_types_word: Dictionary of data types of word-level preprocessed data,
            with entries structured as column name : data type.
        out_types_sentence: Dictionary of data types of sentence-level preprocessed data,
            with entries structured as column name : data type.
        nan_cols: List of column names for columns that can possibly include NaN values.
        fillna: Specifies the fill-NaN strategy enacted during aggregation in get_word_data and get_sentence_data. Default: zero.
            Choose one among:
                - none: leaves NaNs as-is.
                - zero: fills NaNs with 0 => missing duration will count as 0 during averaging.
                - (min|mean|max)_participant: fills NaNs with the min|mean|max value for that token across participants.
            To be added in the future:
                - (min|mean|max)_type: fills NaNs with the min|mean|max value for that token in the whole dataset.
        """
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, data_filename)
        self.out_preprocessed = os.path.join(data_dir, out_filename)
        self.out_cleaned = os.path.join(data_dir, f"fillna_{fillna}_{out_filename}")
        self.text_columns = text_columns
        self.ref_participant = ref_participant
        self.out_types_word = out_types_word
        self.out_types_sentence = out_types_sentence
        self.preprocessed_data = None
        if not os.path.exists(self.out_preprocessed):
            logger.info("Preprocessing dataset, this may take some time...")
            self.create_preprocessed_dataset()
            logger.info("Done preprocessing.")
        logger.info(f"Loading preprocessed data from {self.out_preprocessed}")
        self.preprocessed_data = read_tsv(self.out_preprocessed)
        if not os.path.exists(self.out_cleaned):
            # We fill missing value following the specified strategy
            logger.info(f"Filling NaN values using strategy: {fillna}")
            self.fill_nan_values(fillna, nan_cols)
            logger.info("Done filling NaNs")
        logger.info(f"Loading cleaned data from {self.out_cleaned}")
        self.cleaned_data = read_tsv(self.out_cleaned)

    def fill_nan_values(self, strategy, columns):
        """ Fills NaNs in self.preprocessed_data based on strategy """
        if strategy == "none":
            df = self.preprocessed_data
        elif strategy == "zero":
            df = self.preprocessed_data.fillna({c: 0 for c in columns})
        elif strategy == "min_participant":
            # Slow for huge dataframes, can probably be improved
            # If a value is null for all participants, it is filled with 0s
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_min, columns)
        elif strategy == "mean_participant":
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_mean, columns)
        elif strategy == "max_participant":
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_max, columns)
        else:
            raise AttributeError(f"Strategy {strategy} not supported yet.")
        df = df.sort_index()
        assert all(
            [x == y for x, y in zip(self.preprocessed_data["word_id"], df["word_id"])]
        ), "Word id order mismatch, fillna misbehaved."
        save_tsv(df, self.out_cleaned)

    def create_preprocessed_dataset(self):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    def get_word_data(self, participant="avg"):
        # By default we average scores across participants
        if participant == "avg":

            # Get the textual words, pos, text_index
            words = []
            pos = []
            text_id = []
            word_id_groups = self.cleaned_data.groupby("word_id", sort=False, as_index=False)
            for name, group in word_id_groups:
                words.append(group["word"].tolist()[0])
                pos.append(group["pos"].tolist()[0])
                text_id.append(group["text_id"].tolist()[0])

            # Compute average for each word across participants
            avg_data = self.cleaned_data.groupby("word_id", sort=False, as_index=False).mean()

            # Add textual columns, participant column is "avg" everywhere
            avg_data["word"] = words
            avg_data["pos"] = pos
            avg_data["text_id"] = text_id
            avg_data["participant"] = len(avg_data) * ["avg"]

            # # Textual fields not considered in avg
            # txt = self.cleaned_data[self.text_columns]
            # # ref_participant has all words annotated, we replace it with 'avg' in the data field
            # txt = txt[txt.participant == self.ref_participant]
            # txt.participant = "avg"
            #
            # # Join numeric and textual fields, remove possible duplicate columns
            # avg_data = pd.concat([txt.reset_index(), scores.reset_index()], axis=1, sort=False)

            # Remove possible duplicate columns
            avg_data = avg_data.loc[:, ~avg_data.columns.duplicated()]
            # Order columns in preprocessed shape
            avg_data = avg_data[list(self.out_types_word.keys())]
            return avg_data
        else:  # Data cleaned with fillna strategy are used for single participants, too
            subset = self.cleaned_data[self.cleaned_data["participant"] == participant]
            if len(subset) == 0:
                raise AttributeError(
                    f"Participant was not found in the dataset. Please choose one"
                    f" among: avg, {', '.join(list(set(self.preprocessed_data['participant'])))}"
                )
            return subset

    def get_sentence_data(self, participant="avg", min_len=-1, max_len=100000):
        # By default we average scores across participants
        data = self.get_word_data(participant)

        # Compute average for each word across participants
        group_sent = data.groupby("sentence_id", sort=False, as_index=False)
        scores = group_sent.sum()

        # Textual fields not considered in avg
        sentences = []
        for name, group in group_sent:
            sentence_tokens = []
            for word in group['word']:
                sentence_tokens.append(str(word))
            sentence = " ".join(sentence_tokens)
            sentences.append(sentence)

        #sentences = list(group_sent["word"].apply(lambda v: " ".join([str(y) for y in v])))

        scores["sentence"] = sentences
        scores["participant"] = [participant for i in range(len(scores))]
        scores["text_id"] = list(group_sent["text_id"].first()["text_id"])

        # Add token counts per sentence using whitespace tokenization
        token_counts = []
        for name, group in group_sent:
            sentence_tokens = []
            for word in group['word']:
                sentence_tokens.append(str(word))
            token_counts.append(len(sentence_tokens))

        #token_counts = list(group_sent["word"].apply(len))
        scores["token_count"] = token_counts

        scores = scores.astype(self.out_types_sentence)
        scores = scores[list(self.out_types_sentence.keys())]

        # Filter based on whitespace token count (not the same as n_token from features!)
        scores = scores[(scores["token_count"] <= max_len) & (scores["token_count"] >= min_len)]

        return scores


class GECOProcessor(EyetrackingProcessor):
    """ Reader that converts the GECO dataset in a preprocessed format. """

    def __init__(self, data_dir, **kwargs):
        # Named after original files from http://expsy.ugent.be/downloads/geco/
        self.materials_path = os.path.join(data_dir, "EnglishMaterials.xlsx")
        # Those are generated files provided in the repository
        self.sentence_ids_path = os.path.join(data_dir, "geco_english_sentence_ids.tsv")
        super(GECOProcessor, self).__init__(
            data_dir,
            data_filename="MonolingualReadingData.xlsx",
            out_filename="preprocessed_geco.tsv",
            ref_participant="pp21",
            **kwargs,
        )

    def create_preprocessed_dataset(self):
        data = pd.read_excel(
            self.data_path, usecols=GECO_DATA_COLS, sheet_name="DATA", na_values=GECO_NA_VALUES, keep_default_na=False,
            engine='openpyxl'
        )
        extra = pd.read_excel(
            self.materials_path, sheet_name="ALL", na_values=["N/A"], keep_default_na=False, usecols=GECO_MATERIAL_COLS,
            engine='openpyxl'
        )
        sent_ids = read_tsv(self.sentence_ids_path)
        logger.info("Preprocessing values for the dataset...")
        df = pd.merge(data, extra, how="left", on="WORD_ID")
        df = pd.merge(df, sent_ids, how="left", on="WORD_ID")
        # Clean up words since we need to rely on whitespaces for aligning
        # sentences with tokens.
        df["WORD"] = [str(w).replace(" ", "") for w in df["WORD"]]

        # Create new fields for the dataset
        text_id = [f"{x}-{y}" for x, y in zip(df["PART"], df["TRIAL"])]
        length = [len(str(x)) for x in df["WORD"]]
        # Handle the case where we don't fill NaN values
        mean_fix_dur = []
        for x, y in zip(df["WORD_TOTAL_READING_TIME"], df["WORD_FIXATION_COUNT"]):
            if pd.isna(x):
                mean_fix_dur.append(np.nan)
            elif y == 0:
                mean_fix_dur.append(0)
            else:
                mean_fix_dur.append(x / y)
        refix_count = [max(x - 1, 0) for x in df["WORD_RUN_COUNT"]]
        reread_prob = [x > 1 for x in df["WORD_FIXATION_COUNT"]]
        # Handle the case where we don't fill NaN values
        tot_regr_from_dur = []
        for x, y in zip(df["WORD_GO_PAST_TIME"], df["WORD_SELECTIVE_GO_PAST_TIME"]):
            if pd.isna(x) or pd.isna(y):
                tot_regr_from_dur.append(np.nan)
            else:
                tot_regr_from_dur.append(max(x - y, 0))
        # 2050 tokens per participant do not have POS info.
        # We use a special UNK token for missing pos tags.
        pos = [GECO_POS_MAP[x] if not pd.isnull(x) else GECO_POS_MAP["UNK"] for x in df["PART_OF_SPEECH"]]
        fix_prob = [1 - x for x in df["WORD_SKIP"]]

        # # fill in nans in sentence ids
        # new_sent_ids = []
        # saved_id = ''
        # for id in df["SENTENCE_ID"].tolist():
        #     if math.isnan(id):
        #         new_sent_ids.append(saved_id)
        #     else:
        #         new_sent_ids.append(id)
        #         saved_id = id

        # Format taken from Hollenstein et al. 2019 "NER at First Sight"
        out = pd.DataFrame(
            {
                # Identifiers
                "participant": df["PP_NR"],
                "text_id": text_id,  # PART-TRIAL for GECO
                "sentence_id": df["SENTENCE_ID"],  # Absolute sentence position for GECO
                #"sentence_id": new_sent_ids,
                # AOI-level measures
                "word_id": df["WORD_ID"],
                "word": df["WORD"],
                "length": length,
                "pos": pos,
                # Basic measures
                "fix_count": df["WORD_FIXATION_COUNT"],
                "fix_prob": fix_prob,
                "mean_fix_dur": mean_fix_dur,
                # Early measures
                "first_fix_dur": df["WORD_FIRST_FIXATION_DURATION"],
                "first_pass_dur": df["WORD_GAZE_DURATION"],
                # Late measures
                "tot_fix_dur": df["WORD_TOTAL_READING_TIME"],
                "refix_count": refix_count,
                "reread_prob": reread_prob,
                # Context measures
                "tot_regr_from_dur": tot_regr_from_dur,
                "n-2_fix_prob": ([0, 0] + fix_prob)[: len(df)],
                "n-1_fix_prob": ([0] + fix_prob)[: len(df)],
                "n+1_fix_prob": (fix_prob + [0])[1:],
                "n+2_fix_prob": (fix_prob + [0, 0])[2:],
                "n-2_fix_dur": ([0, 0] + list(df["WORD_TOTAL_READING_TIME"]))[: len(df)],
                "n-1_fix_dur": ([0] + list(df["WORD_TOTAL_READING_TIME"]))[: len(df)],
                "n+1_fix_dur": (list(df["WORD_TOTAL_READING_TIME"]) + [0])[1:],
                "n+2_fix_dur": (list(df["WORD_TOTAL_READING_TIME"]) + [0, 0])[2:],
            }
        )
        # Convert to correct data types
        out = out.astype(self.out_types_word)
        # Caching preprocessed dataset for next Processor calls
        save_tsv(out, self.out_preprocessed)
        logger.info(f"GECO data were preprocessed and saved as" f" {self.out_preprocessed} with shape {out.shape}")
        self.preprocessed_data = out

    def get_sentence_data(self, participant="avg", min_len=5, max_len=45):
        scores = super(GECOProcessor, self).get_sentence_data(participant)
        # Filter based on whitespace token count (not the same as n_tokens from features!)
        scores = scores[(scores["token_count"] <= max_len) & (scores["token_count"] >= min_len)]
        # Order columns in preprocessed shape
        return scores


def fillna_min(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].min().fillna(0))})


def fillna_mean(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].mean().fillna(0))})


def fillna_max(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].max().fillna(0))})
