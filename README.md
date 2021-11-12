## CS-401 Applied Data Analysis - Project
This repository contains a data analysis project for CS-401 Applied Data Analysis at EPFL.

### Authors
- Maciej Styczeń - maciej.styczen@epfl.ch
- Nicolo de Sabbata - camillo.desabbata@epfl.ch
- Dixit Sabharwal - dixit.sabharwal@epfl.ch
- Elia Fantini - elia.fantini@epfl.ch

## You are what you say: can a bunch of words tell what your political view is?

## Abstract
Political opinions can be one of the most socially unifying or divisive topics, defining the people we surround ourselves with. It has been strongly debated the extent to which contemporary political parties well represent each ideology and their internal and temporal coherence of opinions. What we plan to investigate in the following project is how such opinions and ideas can influence the way politicians speak, what they talk about, and the way they do it.
Starting from quotes of US representatives of the Democrats and Republican parties, we divide them by treated subject and perform  sentiment analysis on them across time, to assess the evolution of opinions of the two groups on different matters. Pairing these results with an analysis of the lexical and syntactical properties of the sentences, we try to build a model capable of predicting the affiliations of the speaker based on his quotes.

## Data Preprocessing 
To perform the analysis, we are interested in selecting only the quotes from politicians and matching them with the political affiliation of the speaker. To obtain such a dataset, we perform the following preprocessing steps:
1. Load the Quotebank dataset.
2. Drop the quotes without an attributed speaker.
3. Load the Wikidata table with metadata about the speakers and select the speakers that are affiliated with the Democratic or Republican party.
4. Perform an inner join between the table containing the quotes (Quotebank) and the table containing (Wikidata).
In addition to the previous points, we used grammatical structure and complexity metrics to analyze quotes and drop the ones with outlier values, as they were related to meaningless quotes.
Finally, we obtained a dataset which associates each quote to the speaker who uttered it and his political affiliation.
In total, we have 17.4 million quotes and when serialized the size of the `pickle` file is 2.93GB. From the Manifesto-Project dataset (https://manifestoproject.wzb.eu) we extracted labeled sentences of the two parties' manifestos over years 2012, 2016, and 2020. 

## Topic Labelling 
Additionally, we used the Manifesto-Project dataset (https://manifestoproject.wzb.eu), which provides sentences of the two parties' manifestos over years 2012, 2016, and 2020, labeled accordingly to the issue they deal with. Using this data should allow us to train the ML model of a topic classifier, to add to each quote the subject matter. Thanks to this classifier we may answer the following questions:
1. How did trends about different world problems evolve from 2015 to 2020?
2. What was each year's main discussed theme? Which one was the most popular among all years? 
3. Are there certain matters which are discussed more by a party than the other? 

## Sentiment Analysis
Labeling the quotations as positive/negative, often referred to as sentiment analysis, introduces additional information about the emotion that is carried by the quotation. Attributing sentiment to quotations can be performed using pre-trained transformer models such as [BERT](https://arxiv.org/abs/1810.04805). Once additional information about quotation sentiment is extracted, we want to answer the following research questions:
1. Is there a difference between the fraction of quotations that are positive/negative between the two parties? What does that say about the general attitude of the party?
2. Are there any issues addressed differently by parties? Do they try to rouse different emotions on the same society's problem?
3. Is there any case of a party changing completely in the attitude towards a specific topic during the time?

## Grammatical Structure and Complexity 
Utilizing the Textstat library we added to each quotes different readability, complexity, and grade level metrics. Thanks to these statistics we might see if there's a  significant difference between speakers from different parties, as well as differences between people with identical affiliations but different backgrounds. By studying the properties of the language used by the representatives we want to answer the following research questions:
1. Do speakers of the two groups use different lexicon? Which one uses the largest vocabulary? 
2. Is there any noticeable difference in the complexity of the sentences? Which one is more readabile?  
3. How did the grammar and correctness of sentences evolve over the year? Is it the same across the two parties?

## Construction of the Classifier 
Combining all the data from the preprocessing, enriched by sentiment analysis, topic classification, and grammatical complexity, is it possible to build a classifier that can predict whether the quoted speaker is affiliated with Republicans or Democrats? If yes, it would mean that there is a strong difference in how people sustaining a certain party speak, feel, and think about the main problems of society.

## Internal Milestones & Timeline
* 26 Nov - 3 Dec: In the first week we plan to train the classifier to do topic labeling on our quotes, perform sentiment analysis on the quotes divided both by year and by topic and make use of the Textstat library to examine the grammar and complexity of the sentences.
* 3 Dec - 10 Dec: In the second week we'll start drawing conclusions on the results obtained in the previous week and train the final classifier.
* 10 Dec - 17 Dec: In the final week the analyze the results of the classifier and polish/perfect the graphs and the overall analysis. 

We didn't set up internal milestones yet as we believe we would work better if they were decided week by week, as the project progresses.
