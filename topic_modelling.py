#!/usr/bin/env python
# coding: utf-8

#importing all the libraries
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from gensim import corpora
import gensim
import nltk
import string
import spacy
import en_core_web_sm

filepath = 'Enter filepath'
filename = 'Enter filename'

dataframe=pd.read_csv(filepath+filename)
print(len(dataframe))

import re
from spacy.lang.en import English
parser = English()

#Creating tokens by removing stopwords, punctuation using SpaCy
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    stop_list=['<','>','</i>','<i>','<b>','</b>','=','<i','<b','</i','</b','<sub>','</sub>','<sub']
    parser.Defaults.stop_words.update(stop_list)
    #print(tokens)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.is_stop==True:
            continue
        elif re.search(r'[a-zA-z0-9]*<\/[a-z]+',str(token)):
            continue
        elif token.is_punct==True:
            continue
        else:
            lda_tokens.append(token.lower_)
    #print(lda_tokens)
    return lda_tokens

#Lemmatizing the tokens

nltk.download('wordnet')
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_lemma2(word):
    lemma = wordnet.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def get_lemma(word):
    lemma = wordnet.morphy(word)
    if lemma is None:
        y=word
    else:
        y=lemma
    x=lemmatizer.lemmatize(y, get_wordnet_pos(word))
    return x


nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
en_stop = set(nltk.corpus.stopwords.words('english'))


# In[6]:


def prepare_text_for_lda(text):
    # convert all words into lower case, split by white space
    #  remove words with less than 4 letters (small words, punctuation)
    #tokens = [token for token in tokens if token not in en_stop]
    tokens = tokenize(str(text))
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [get_lemma(token) for token in tokens]
    #print(tokens)
    return tokens

titles = []
abstracts=[]
abstract_text_data=[]
title_text_data=[]
for index, row in dataframe.iterrows():
    #Preparing abstract text for LDA
    abstract = row['Abstract']
    abstracts.append(abstract)
    tokens = prepare_text_for_lda(abstract)
    abstract_text_data.append(tokens)

    #Preparing title text for LDA
    title= row['Title']
    titles.append(title)
    title_tokens=prepare_text_for_lda(title)
    title_text_data.append(title_tokens)
    #print(index)
    if(index==299):
        print("BREAK")
        break

print(len(title_text_data))
print(len(abstract_text_data))


# Creating LDA model for abstracts

# Creating corpus and dicitionary from abstract data

from gensim import corpora
dictionary = corpora.Dictionary(abstract_text_data)
#print(dictionary)
corpus = [dictionary.doc2bow(text) for text in abstract_text_data]
import pickle
#pickle.dump(corpus, open(filepath+filename+'_corpus.pkl', 'wb'))
#dictionary.save(filepath+filename+'_dictionary.gensim')
print(len(corpus))

# Forming the base LDA model


NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary,passes=10)
#ldamodel.save(filepath+filename+'_abstract.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)


# Coherence score for base LDA model
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=ldamodel, texts=abstract_text_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


best_ldamodel = gensim.models.LdaMulticore(corpus,num_topics=7,
                                           random_state=100,
                                           passes=10,
                                           alpha=0.90,
                                           eta=0.61,id2word=dictionary)
#ldamodel.save(filepath+filename+'_abstract.gensim')
best_topics = best_ldamodel.print_topics(num_words=5)
for topic in best_topics:
    print(topic)


from gensim.models import CoherenceModel# Compute Coherence Score
best_coherence_model_lda = CoherenceModel(model=best_ldamodel, texts=abstract_text_data, dictionary=dictionary, coherence='c_v')
best_coherence_lda = best_coherence_model_lda.get_coherence()
print('\nCoherence Score: ', best_coherence_lda)

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(best_ldamodel, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display, filepath+'abstract_anemia_pain_opioid.html')
pyLDAvis.display(lda_display)

# Coherence scores for various values of a,b and k (Could take upto half an hour or more to run)

from gensim.models import CoherenceModel

def compute_coherence_values(corpus,data, dictionary, k, a, b):

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence()


# In[ ]:


import numpy as np
import tqdm
grid = {}
grid['Validation_Set'] = {}# Topics range
min_topics = 5
max_topics = 9
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')


# Validation sets
#num_of_docs = len(corpus)
#corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
               #gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75),
               #corpus]
#corpus_title = ['75% Corpus', '100% Corpus']
model_results = {#'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
}
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=120)
    # iterate through validation corpuses
    #for i in range(len(corpus_sets)):
        # iterate through number of topics
    for k in topics_range:
            # iterate through alpha values
        for a in alpha:
                # iterare through beta values
            for b in beta:
                    # get the coherence score for the given parameters
                cv = compute_coherence_values(corpus=corpus,data=abstract_text_data, dictionary=dictionary,k=k, a=a, b=b)
                    # Save the model results
                   # model_results['Validation_Set'].append(corpus_title[i])
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)

                pbar.update(1)
    #pd.DataFrame(model_results).to_csv('lda_tuning_query_and_or_opioid.csv', index=False)
    pbar.close()


# In[ ]:


pd.DataFrame(model_results).to_csv(filepath+'abstract_query4_and_or_opioid_0.3.csv', index=False)


# ## Creating LDA model for titles

# In[19]:


from gensim import corpora
dictionary_title = corpora.Dictionary(title_text_data)
corpus_title = [dictionary_title.doc2bow(text) for text in title_text_data]
import pickle
#pickle.dump(corpus_title, open(filepath+filename+'_corpus_title.pkl', 'wb'))
#dictionary.save(filepath+filename+'_dictionary_title.gensim')

NUM_TOPICS = 5
ldamodel_title = gensim.models.ldamodel.LdaModel(corpus_title, num_topics = NUM_TOPICS, id2word=dictionary_title, passes=10)
#ldamodel.save(filename+'_title.gensim')
topics = ldamodel_title.print_topics(num_words=5)
for topic in topics:
    print(topic)

from gensim.models import CoherenceModel# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel_title, texts=title_text_data, dictionary=dictionary_title, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


best_ldamodel_title = gensim.models.LdaMulticore(corpus_title,num_topics=7,
                                           random_state=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.90,id2word=dictionary_title)
#ldamodel.save(filepath+filename+'_abstract.gensim')
best_topics = best_ldamodel_title.print_topics(num_words=5)
for topic in best_topics:
    print(topic)

from gensim.models import CoherenceModel# Compute Coherence Score
best_coherence_model_lda_title = CoherenceModel(model=best_ldamodel_title, texts=title_text_data, dictionary=dictionary_title, coherence='c_v')
best_coherence_lda_title = best_coherence_model_lda_title.get_coherence()
print('\nCoherence Score: ', best_coherence_lda_title)

import pyLDAvis.gensim
lda_display_title = pyLDAvis.gensim.prepare(best_ldamodel_title, corpus_title, dictionary_title, sort_topics=False)
pyLDAvis.save_html(lda_display_title, filepath+'title_nicotine_pain_opioid.html')
pyLDAvis.display(lda_display_title)

model_results_titles = {#'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
}# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=120)

    # iterate through validation corpuses
    #for i in range(len(corpus_sets)):
        # iterate through number of topics
    for k in topics_range:
            # iterate through alpha values
        for a in alpha:
                # iterare through beta values
            for b in beta:
                    # get the coherence score for the given parameters
                cv = compute_coherence_values(corpus=corpus_title,data=title_text_data, dictionary=dictionary_title,k=k, a=a, b=b)
                    # Save the model results
                   # model_results['Validation_Set'].append(corpus_title[i])
                model_results_titles['Topics'].append(k)
                model_results_titles['Alpha'].append(a)
                model_results_titles['Beta'].append(b)
                model_results_titles['Coherence'].append(cv)

                pbar.update(1)
    #pd.DataFrame(model_results).to_csv('lda_tuning_query_and_or_opioid.csv', index=False)
    pbar.close()

pd.DataFrame(model_results_titles).to_csv(filepath+'Enter filename', index=False)
