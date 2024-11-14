from glob import glob

import pandas as pd
import streamlit as st

from imports.game_class import Game
from imports.my_logger import logger
from imports.preprocess_data import (
    add_label_to_df,
    apply_for_clean_text,
    apply_for_tok_lemma_tex,
    treat_NaN_req_df,
)
from imports.request_reviews import reviews_df_task
from imports.save_load_data import (
    cloudpkl_game_dump,
    cloudpkl_game_load,
)
from imports.visualization_exploration import (
    plot_label_distribution,
    plot_wordcloud,
    tf_summary,
)

df_summary: pd.DataFrame = pd.DataFrame()


def pre_ml_task(
    appid: str,
    game_name: str = "noname",
    load_game: bool = True,
    save_game: bool = False,
    new_request: bool = False,
) -> Game:
    new_game: bool = not load_game
    # Create or load a game of class Game
    if load_game and glob(f"./saved_data/{appid}.pkl"):
        load = cloudpkl_game_load("%s" % appid)
        if isinstance(load, Game):
            game: Game = load
            st.success("Game object with appid= %s loaded!" % appid)
        else:
            game: Game = Game(game_name, appid)
            new_game = True
            logger.warning(
                "Couldn't load the game object with appid= %s from ./saved_data/%s.pkl\nCreated new game object instead.",
                appid,
                appid,
            )
            st.warning(
                "Couldn't load the Game object with appid= %s\nCreated new game object instead."
                % appid
            )
    else:
        game = Game(game_name, appid)
        new_game = True

    if new_request | new_game:
        with st.spinner("Downloading Steam Reviews..."):
            game = reviews_df_task(game)
        st.success(
            f"Reviews aquired!\nDataFrame for game '{game.name}' has:\nReviews={game.df.shape[0]} and Columns={game.df.shape[1]}"
        )

    # Load or create the processed dataframe
    if new_game:
        # Treats NaN values
        game = treat_NaN_req_df(game)

        # Add label to df
        game = add_label_to_df(game)

        # Clean the text and populate game.df
        game = apply_for_clean_text(game)

        # Tokenize and lemmatize the text and populate game.df
        game = apply_for_tok_lemma_tex(game)

    # Save the game object
    if save_game:
        cloudpkl_game_dump(game)
        st.success("Game with appid= %s saved!" % appid)

    # Display a sample of req_df
    st.subheader(
        "DataFrame from API response",
        "Requested_DataFrame",
        help="With some columns dropped\n",
    )
    st.write(game.req_df.sample(4))

    # Display a sample of df
    st.subheader(
        "Preprocessed DataFrame\n",
        "Preprocessed_DataFrame",
        help="The DataFrame with the text cleaned and tokenized and lemmatized",
    )
    st.write(game.df.sample(4))

    # Plot label distribution
    st.subheader("Label Distribution", "Label_Distribution")
    plot_label_distribution(game.df)

    # Generates the term frequency and analyzes of the top 20
    st.subheader("Term Frequency Summary", "TermFrequency_Summary")
    _ = tf_summary(game)

    # Plot WordCloud of the top 100 words
    st.subheader("WordCloud", "WordCloud", help="WordCloud of the top 100 words")
    plot_wordcloud(game)

    return game
