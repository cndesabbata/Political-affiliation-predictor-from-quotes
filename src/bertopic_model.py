# before running this code,check the following link,
# otherwise the notebook will probably crash due to insufficient ram
# https://maartengr.github.io/BERTopic/faq.html#i-am-facing-memory-issues-help


from math import ceil
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from helpers import expand_contractions, remove_words, preprocess_quote
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic

def train_bertopic():
    #  load pickle of quotebank
    QUOTEBANK_PATH = "../data/binary/us-politicians.pickle"
    data = pd.read_pickle(QUOTEBANK_PATH)
    data = data.sample(600000)
    #  extract quotations
    quotes = data['quotation']
    # expand contractions
    quotes = quotes.apply(lambda quote: expand_contractions(quote))

    # load the stopwords (extend the standard list by the contraction leftovers) and remove them from the quotes
    stop_words = stopwords.words('english') + ['nt', 'ca', 'wo']
    quotes = quotes.apply(lambda quote: remove_words(quote, stop_words))

    # apply other preprocessing
    quotes = quotes.apply(lambda quote: preprocess_quote(quote))

    # look at most common words
    tokenized = quotes.apply(lambda quote: word_tokenize(quote))
    words = tokenized.explode()
    words = words.astype("str")
    words = words.value_counts()[:100]
    words = words.keys()
    # Stopwords and special characters removal
    data['quotation'] = data['quotation'].str.replace('\W',' ')
    data['quotation'] = data['quotation'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (words)]))
    # extract preprocessed quotations from dataset
    quotes = list(data['quotation'])
    # some tricks to prevent insufficient RAM
    del data
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)
    topic_model = BERTopic(verbose= True, vectorizer_model=vectorizer_model, min_topic_size=50, n_gram_range=(1,3), low_memory=True, calculate_probabilities=False, nr_topics= 50)
    # training the model
    topics = topic_model.fit(quotes)

    freq = topic_model.get_topic_info()
    topics = freq['Topic']
    # saving found topics and probabilities in a txt file
    file = open("topics.txt","w")
    for topic in topics:
        file.write("Topic "+str(topic)+": ")
        output = topic_model.get_topic(topic)
        file.write(str(output))
        file.write("\n")
    file.close()
    #  saving the model
    topic_model.save("BERTopic_model")

    quotes = pd.read_pickle(QUOTEBANK_PATH)
    # applying the same preprocessing to new quotes
    quotes['quotation'] = quotes['quotation'].str.replace('\W',' ')
    quotes['quotation'] = quotes['quotation'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (words)]))
    quotes = list(quotes['quotation'])
    len = len(quotes)
    part_size = ceil(len/4)
    all_predictions=[]
    # for RAM limitations, computing predictions on new data dividing the workload into 4 parts
    for i in range(4):
            predictions, probabilities = topic_model.transform(quotes[(i*part_size) : (i+1)*part_size])
            all_predictions.extend(predictions)
    topic_model = None  # to free memory
    del quotes  # to free memory
    df = pd.read_pickle(QUOTEBANK_PATH)
    df['cluster'] = all_predictions
    #  saving the result into a pickle
    df.to_pickle('../data/binary/us-politicians-with-clusters.pickle')


