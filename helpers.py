# imports
import textstat as ts
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
import bz2
from statistics import mode
from datetime import datetime
import matplotlib.pyplot as plt
import re
import string
import contractions
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def generate_speaker_affiliations(parquet_path, out_path, remove_raw=False):

    # load speaker info
    speaker_info = pd.read_parquet(parquet_path)
    speaker_info = speaker_info[["id", "label", "party"]]

    # take the speakers that have an assigned political affiliation
    speaker_info = speaker_info.dropna()

    # in case of multiple affiliations, take the first affiliation only
    # speaker_info["party"] = speaker_info["party"].apply(lambda x: int(x[0][1:]))

    # alternatively (I think a slightly better way), select most common party
    speaker_info["party"] = speaker_info["party"].apply(lambda x: mode(x)[1:])

    # transform speaker id into int
    speaker_info["id"] = speaker_info["id"].apply(lambda x: int(x[1:]))

    print(f"Speaker affiliation DF:\n {speaker_info.head()}")

    speaker_info.to_pickle(out_path)

    if remove_raw:
        os.remove(parquet_path)


def save_pickle(json_path_bz2, pickle_path, remove_raw=False):
    data = []
    with bz2.open(json_path_bz2, 'rb') as s_file:
        print("Quotation file opened...")
        for instance in tqdm(s_file):
            instance = json.loads(instance)  # loading a sample

            # if there is no speaker, skip current row
            if not instance['qids']:
                continue

            # else proceed to read the data
            row = dict()
            row['speaker_id'] = int(instance['qids'][0][1:])
            row['quote_id'] = instance['quoteID']
            row['quotation'] = instance['quotation']
            data.append(row)

        df = pd.DataFrame(data)
        df.to_pickle(pickle_path)

    if remove_raw:
        os.remove(json_path_bz2)


def join_quotes_with_speaker_affiliations(df_quotes, df_affiliations, out_path):
    # join the quote data with their corresponding labels
    merged = pd.merge(left=df_quotes, left_on="speaker_id",
                      right=df_affiliations, right_on="id")
    merged = merged.drop(columns=["id"])
    merged = merged.rename(columns={"label": "speaker"})
    print(f"Merged DF: \n{merged.head()}")
    merged.to_pickle(out_path)


def plot_across_time(democrats_across_time, republicans_across_time):
    # Change the date format and count number of quotes per month
    democrats_across_time["Date-Time"] = democrats_across_time["Date-Time"].dt.strftime(
        '%Y-%m')
    republicans_across_time["Date-Time"] = republicans_across_time["Date-Time"].dt.strftime(
        '%Y-%m')
    democrats_across_time = democrats_across_time.groupby(
        ['Date-Time'], as_index=False).agg('count')
    republicans_across_time = republicans_across_time.groupby(
        ['Date-Time'], as_index=False).agg('count')

    # Plot the obtained results
    fig, ax = plt.subplots()
    ax.bar(democrats_across_time['Date-Time'], democrats_across_time['count'],
           log=True, alpha=0.5, label="Democrats", align='center')
    ax.bar(republicans_across_time['Date-Time'], republicans_across_time['count'],
           log=True, alpha=0.5, label="Republicans", align='center')
    plt.xticks(np.arange(0, len(democrats_across_time['Date-Time']) + 1, 12))
    ax.set_xlabel('Date')
    ax.set_ylabel('Num. of quotations')
    plt.show()


def plot_length(democrats, republicans):
    # Plot the distribution of quotes' length
    democrats_lengths = democrats["quotation"].apply(lambda x: len(x))
    republicans_lengths = republicans["quotation"].apply(lambda x: len(x))

    plt.hist(democrats_lengths, log=True,
             alpha=0.5, label="Democrats", bins=100)
    plt.hist(republicans_lengths, log=True,
             alpha=0.5, label="Republicans", bins=100)
    plt.axvline(republicans_lengths.mean(), color='red',
                label=f"mean republicans: {np.round(republicans_lengths.mean(), 1)}")
    plt.axvline(democrats_lengths.mean(), color='blue',
                label=f"mean democrats: {np.round(democrats_lengths.mean(), 1)}")
    plt.axvline(republicans_lengths.median(), color='yellow',
                label=f"median republicans: {np.round(republicans_lengths.median(), 1)}")
    plt.axvline(democrats_lengths.median(), color='green',
                label=f"median democrats: {np.round(democrats_lengths.median(), 1)}")
    plt.xlabel('Length of quotation')
    plt.ylabel('Num. of quotations')
    plt.title("Length of politicans' quotations.")
    plt.legend()
    plt.show()


def plot_metrics_scores(dem_data, rep_data, title):
    # plotting distribution, mean and median for both republicans and democrats, in the four calculated scores
    metrics_columns = ['flesch_reading_ease',
                       'dale_chall_readability_score', 'text_standard', 'reading_time']
    fig, axs = plt.subplots(4, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)
    bins = np.logspace(0, np.log10(10**3), 50)

    axs[0].hist(dem_data['flesch_reading_ease'],
                log=True, alpha=0.5, bins=bins)
    axs[0].hist(rep_data['flesch_reading_ease'],
                log=True, alpha=0.5, bins=bins)
    axs[0].set_title("Flesch reading ease, higher is easier")
    axs[1].hist(dem_data['dale_chall_readability_score'],
                log=True, alpha=0.5, bins=bins)
    axs[1].hist(rep_data['dale_chall_readability_score'],
                log=True, alpha=0.5, bins=bins)
    axs[1].set_title("Dale Chall readability score, lower is easier")
    axs[2].hist(dem_data['text_standard'], log=True, alpha=0.5, bins=bins)
    axs[2].hist(rep_data['text_standard'], log=True, alpha=0.5, bins=bins)
    axs[2].set_title("Text Standard test score, lower is easier")
    axs[3].hist(dem_data['reading_time'], log=True, alpha=0.5, bins=bins)
    axs[3].hist(rep_data['reading_time'], log=True, alpha=0.5, bins=bins)
    axs[3].set_title("Reading time")
    metrics_rep = rep_data[metrics_columns]
    metrics_dem = dem_data[metrics_columns]
    mean_dem = metrics_dem.mean(axis=0)
    median_dem = metrics_dem.median(axis=0)
    mean_rep = metrics_rep.mean(axis=0)
    median_rep = metrics_rep.median(axis=0)

    for i, ax in enumerate(axs):
        ax.axvline(mean_rep[i], color='red',
                   label=f"mean republicans: {np.round(mean_rep[i], 1)}")
        ax.axvline(mean_dem[i], color='blue',
                   label=f"mean democrats: {np.round(mean_dem[i], 1)}")
        ax.axvline(median_rep[i], color='yellow',
                   label=f"median republicans: {np.round(median_rep[i], 1)}")
        ax.axvline(median_dem[i], color='green',
                   label=f"median democrats: {np.round(median_dem[i], 1)}")
        ax.legend()
        ax.set_xscale('log')
    plt.show()


def preprocess_quote(quote):
    # to lowercase
    quote = quote.lower()

    # remove numbers and punctuation
    quote = re.sub(r'\d+', '', quote)
    quote = quote.translate(str.maketrans('', '', string.punctuation))

    # remove all single characters
    quote = re.sub(r'\s+[a-zA-Z]\s+', ' ', quote)
    # Remove single characters from the start
    quote = re.sub(r'\^[a-zA-Z]\s+', ' ', quote)

    # remove leading, trailing, and repeating spaces
    quote = re.sub(' +', ' ', quote)
    quote = quote.strip()

    return quote


def expand_contractions(quote):
    return contractions.fix(quote)


def remove_words(quote, words, stem=False):
    tokens = word_tokenize(quote)
    if stem:
        words = [PorterStemmer().stem(word) for word in words]
        filtered_tokens = [token for token in tokens if PorterStemmer().stem(
            token).lower() not in words]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in words]
    return " ".join(filtered_tokens)


def read_most_common(path):
    most_common = []
    with open(path, "r") as file:
        for line in file.readlines():
            most_common.append(line.strip())
    return most_common


def create_model_topic_clusterer(quotes):
    # some tricks to prevent insufficient RAM
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words="english", min_df=10)
    topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model, min_topic_size=50, n_gram_range=(
        1, 3), low_memory=True, calculate_probabilities=False)
    topics, probs = topic_model.fit_transform(quotes)

    freq = topic_model.get_topic_info()

    topic_model.save("data/BERTopic_model")
