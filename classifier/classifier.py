import numpy as np
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def preprocess_sentences(X):
    documents = []
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)

    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    return X


def create_topic_classifier(X = list,y = np.array, SAVE_MODEL_PATH = str):
    X = preprocess_sentences(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    with open(SAVE_MODEL_PATH, 'wb') as picklefile:
        pickle.dump(classifier,picklefile)



if __name__ == "__main__":
    DATASET_PATH = 'data/join_result.csv'
    df = pd.read_csv(DATASET_PATH)
    X = list(df['sentence'])
    y = np.array(df['label'])
    create_topic_classifier(X, y, 'topic_classifier')
