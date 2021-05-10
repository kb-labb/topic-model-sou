import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import re
import os
import pickle
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import stanza
from tqdm import tqdm

stanza.download('sv')

df = pd.read_csv("sou_1995-1999_pages.csv")
df = df.drop("Unnamed: 0", axis=1)
df = df.dropna()

path = "models/1995-1999/"

"""
# use this part if you want to do a department-based topic model

df_dep = pd.read_csv("all_departments.csv")
df_for = df_dep[df_dep['department'] == "försvarsdepartementet"]
path = "models/försvarsdepartementet/"

os.mkdir(path)

files = os.listdir("dataset_pages")
list_of_dfs = []

for file in files:
    df_pages = pd.read_csv("dataset_pages/"+file)
    df_pages = df_pages.drop("Unnamed: 0", axis=1)
    list_of_dfs.append(df_pages.merge(df_for, on=['year','issue']))

df = pd.concat(list_of_dfs)
df = df.dropna()
"""

print("Preprocessing dataframe...")

# turn all whitespace characters into one space
df['text'] = df['text'].map(lambda x: re.sub(r'\s+', ' ', x))

# this is necessary because the rows of dots break the sentence segmentation
df['text'] = df['text'].map(lambda x: re.sub(r'\.+', '.', x))

# a step to put parts of words separated by hyphen and space back together
df['text'] = df['text'].map(lambda x: re.sub(r'- (?![och|eller])', '', x))

# remove numbers
df['text'] = df['text'].map(lambda x: re.sub(r'\d+', '', x))

# only keep rows with at least 20 words
df = df[df['text'].str.split().str.len() >= 20]

print("...Done")

pages = list(df.text)

# load stanza model
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma')
allowed_tags=['NOUN', 'VERB']

with open("stopwords.txt", 'r') as f:
    stopwords = [word.strip() for word in f.readlines()]

stopwords.extend(["ha", "finnas", "finnes", "andel", "procent", "lag", "utredning", "myndighet", "kap.",
                  "förslag", "bestämmelse", "åtgärd", "fråga", "år", "se", "innebära", "beslut", "te", "ra",
                  "st", "dr", "fe", "na", "rr", "le", "sid.", "exkl.", "bedömning", "kapitel", "prop.",
                  "lagstiftning", "regering"])

# stanza does sentence segmentation by default
# by taking the lemma we also automatically lowercase the words
data = []

print("Removing stopwords, punctuation and lemmatizing...")

bad_pages = []
for page in tqdm(pages):
    try:
        doc = nlp(page)
        data.append([word.lemma for sent in doc.sentences for word in sent.words if word.pos in allowed_tags
                    and word.lemma not in stopwords and re.search(r'\w{2}', word.lemma) is not None])
    except:
        bad_pages.append(page)
        continue

with open(path+"bad_pages.txt", "a") as f:
    f.writelines(bad_pages)

pd.DataFrame(data).to_csv(path+"data.csv",index=False)

# create the dictionary
print("Creating the dictionary...")
id2word = corpora.Dictionary(data)
print('Total Vocabulary Size before cleaning:', len(id2word))

# keep only words that appear in at least 10 documents but less than 50% of all documents
id2word.filter_extremes(no_below=10, no_above=0.5)
print('Total Vocabulary Size after cleaning:', len(id2word))

id2word.save(path+"id2word.dict")
print("Dictionary saved to file.")

# create final corpus
corpus = [id2word.doc2bow(text) for text in data]
pd.DataFrame(corpus).to_csv(path+"corpus.csv",index=False)
print("Final corpus created and saved to file.")

mallet_path = "Mallet/bin/mallet"

print("Starting evaluation of number of topics...")
coherence = []
for k in range(15,60,5):
    print('Topics: '+str(k))
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=k, id2word=id2word)

    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    coherence.append((k,coherence_ldamallet))

x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]

plt.plot(x_val,y_val)
plt.scatter(x_val,y_val)
plt.title('Number of Topics vs. Coherence')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence')
plt.xticks(x_val)
plt.savefig(path+'number_of_topics.png')

max_index = y_val.index(max(y_val))
n_topics = x_val[max_index]

print(f"Building best topic model with {n_topics} topics...")

final_tm = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=n_topics, id2word=id2word)
pickle.dump(final_tm, open(path+"best_tm.pkl", "wb"))

print("Best topic model saved.")
