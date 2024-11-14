import re
from string import punctuation

import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.util import is_cjk

from imports.game_class import Game
from imports.my_logger import logger


def treat_NaN_req_df(g: Game) -> Game:
    """Checks if there are NaN values in the DataFrame and attemps to fix them inplace

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        bool: True if there still are NaN values, False otherwise
    """

    def _remove_NaN(df: pd.DataFrame) -> bool:
        """Attemps to fixes NaN values inplace in the DataFrame.

        If NaN values are found in 'author_playtime_at_review', they are replaced with 0.0
        If Nan values are found in 'review', the entry is dropped

        Args:
            df (pd.DataFrame): The DataFrame with the NaN values

        Returns:
            bool: False if there still are NaN values, True otherwise
        """
        values = {"author_playtime_at_review": 0.0}
        df.fillna(values, inplace=True)
        df.dropna(subset=["review"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return not (df.isna().any().any())

    df: pd.DataFrame = g.req_df
    nan_count = df.isna().sum().sum()
    if not nan_count:
        logger.info("DataFrame for game '%s' has no NaN values.", df.columns.name)
        return g
    if _remove_NaN(df):
        logger.info(
            "DataFrame for game '%s' had %d NaN values, they have been resolved.",
            df.columns.name,
            nan_count,
        )
        st.write(
            f"DataFrame for game '{g.name}' had {nan_count} NaN values, they have been resolved."
        )
        return g

    df_nan_mask = df.isna()
    text: str = f"DataFrame for game '{g.name}' has {df_nan_mask.sum().sum()} NaN values:\n{df_nan_mask.sum().loc[df_nan_mask.sum() > 0].sort_values(ascending=False).to_dict()}"
    logger.warning(text)
    st.write(text)
    st.write(df.loc[df_nan_mask.any(axis=1)])
    return g


def add_label_to_df(g: Game) -> Game:
    g.df["label"] = g.req_df["voted_up"].astype(int)
    return g


stopwords_en_edited: set[str] = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "between",
    "into",
    "through",
    "to",
    "from",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "any",
    "both",
    "each",
    "other",
    "such",
    "own",
    "so",
    "than",
    "too",
    "can",
    "will",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "wo",
    "won't",
    "wouldn",
    "wouldn't",
    "in",
    "out",
}  # ? perhaps add 'like'


def clean_text(text: str) -> str:
    # Lower case
    text = text.lower()

    # Contract repetitions (>2)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Decontract
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"[\'`´]re", " are", text)
    text = re.sub(r"[\'`´]s", " is", text)
    text = re.sub(r"[\'`´]d", " would", text)
    text = re.sub(r"[\'`´]ll", " will", text)
    text = re.sub(r"[\'`´]ve", " have", text)
    text = re.sub(r"[\'`´]m", " am", text)

    # Strip markdown
    text = re.sub(r"\[.*?\]", "", text)

    # Define punctuation
    punct: str = re.sub(r"\?", "", punctuation + "‘’…“”⠄´")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", punct))

    return text


def tok_lemma_text(text: str, lang: str = "english") -> list[str] | None:
    lemmatizer = WordNetLemmatizer()
    tokenizer = word_tokenize

    # Tokenize
    tokens: list[str] = tokenizer(text, language=lang)

    # Remove Stopwords
    tokens = [tok for tok in tokens if tok not in stopwords_en_edited]

    # Lemmatize and remove cjk chars
    cjk_flag: bool = False
    for i, tok in enumerate(tokens):
        if is_cjk(tok[0]):
            tokens[i] = ""
            cjk_flag = True
        else:
            tok = lemmatizer.lemmatize(tok, "n")
            tok = lemmatizer.lemmatize(tok, "v")
            tok = lemmatizer.lemmatize(tok, "a")
            tok = lemmatizer.lemmatize(tok, "r")
            tokens[i] = tok

    # Remove empty strings
    if cjk_flag:
        tokens = [t for t in tokens if t]

    # Return the tokens list or None if empty list
    return tokens if tokens else None


def apply_for_clean_text(g: Game) -> Game:
    g.df["review"] = g.req_df["review"].apply(clean_text)
    return g


def apply_for_tok_lemma_tex(g: Game) -> Game:
    g.df["words"] = g.df["review"].apply(tok_lemma_text, lang=g.get_params("language"))
    g.df.dropna(inplace=True)
    g.df.reset_index(drop=True, inplace=True)
    return g
