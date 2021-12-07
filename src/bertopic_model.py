# before running this code,check the following link,
# otherwise the notebook will probably crash due to insufficient ram
# https://maartengr.github.io/BERTopic/faq.html#i-am-facing-memory-issues-help
# Create Topic Model

import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#load pickle of quotebank
QUOTEBANK_PATH = "../data/binary/us-politicians.pickle"
data = pd.read_pickle(QUOTEBANK_PATH)
data = data.sample(800000)

# Stopwords and special characters removal
data['quotation'] = data['quotation'].str.replace('\W',' ')
stop = stopwords.words('english')
data['quotation'] = data['quotation'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
# extract preprocessed quotations from dataset
quotes = list(data['quotation'])

# some tricks to prevent insufficient RAM
del data
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)
topic_model = BERTopic(verbose= True, vectorizer_model=vectorizer_model, min_topic_size=50, n_gram_range=(1,3), low_memory=True, calculate_probabilities=False)
topics, probs = topic_model.fit_transform(quotes)

freq = topic_model.get_topic_info()
print(freq.head(21))

topic_model.save("BERTopic_model")
