import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk import FreqDist
from wordcloud import WordCloud

from imports.game_class import Game


def plot_label_distribution(df: pd.DataFrame, x: str = "label") -> None:
    """plot_target_distribution Plots the distributions of the target variable

    Args:
        df (pd.DataFrame): The DataFrame with the target variable

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, layout="constrained", facecolor="lightgrey")

    sns.countplot(data=df, x=x, hue=x, stat="count", legend=False, ax=axs[0])
    axs[0].set(xlabel="", ylabel="Count", title="Count")

    sns.countplot(data=df, x=x, hue=x, stat="percent", legend=False, ax=axs[1])
    axs[1].set(xlabel="", ylabel="Percentage (%)", title="Percentage")

    fig.suptitle(f"Game: {df.columns.name}")
    fig.supxlabel(f"Reviews (total ={df.shape[0]:,d})")
    st.write("Label Distribuition:\n", fig)


def tf_summary(g: Game) -> FreqDist:
    """tf_summary Generates a frequency distribution for the words in the reviews. And analyzes the top 20 most common words.

    Args:
        g (Game): Game object

    Returns:
        FreqDist: Frequency distribution of the words in the reviews
    """
    fd = FreqDist(g.df["words"].explode().to_list())
    cumsum = 0

    for word in fd.most_common(20):
        cumsum += word[1]

    st.write(f"Top 20 most common words in reviews for '{g.name}':")

    fd.tabulate(20)

    st.write(
        f"This top 20 represents {cumsum*100/fd.N():.2f}% or {cumsum:,d} out of the {fd.N():,d} total frequencies.\n\nWhile these words represents {2000/fd.B():.2f}% out of the total {fd.B():,d} unique words."
    )

    fig = plt.figure(facecolor="lightgrey")
    ax = fd.plot(
        20, title=f"Top 20 most common words in reviews for '{g.name}'", show=True
    )
    fig.add_axes(ax)
    st.pyplot(fig)

    return fd


def plot_wordcloud(g: Game, bg: str = "black") -> None:
    wc = WordCloud(
        max_font_size=100,
        max_words=100,
        background_color="black",
        scale=10,
        width=800,
        height=400,
    ).generate("".join([" ".join(w) for w in g.df.words]))

    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wc)
    plt.axis("off")
    st.write(f"Wordcloud for '{g.name}'\n", fig)
