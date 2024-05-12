import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

dataset = pd.read_csv('papers_dataset.csv').iloc[400:]
print(dataset.shape)
dataset.keywords = dataset.keywords.apply(ast.literal_eval)
dataset.sections = dataset.sections.apply(ast.literal_eval)

all_sentences = []
categories = []
titles = []

for row in tqdm(dataset.iterrows()):
    cat = row[1]['category']
    title = row[1]['title']
    
    abstract_sents = sent_tokenize(row[1]['abstract'])
    all_sentences += abstract_sents
    categories += [cat] * len(abstract_sents)
    titles += [title] * len(abstract_sents)
    
    sections = list(row[1]['sections'].values())
    for section in sections:
        section_sents = sent_tokenize(section)
        all_sentences += section_sents
        categories += [cat] * len(section_sents)
        titles += [title] * len(section_sents)
        
print('Number of all sentences:', len(all_sentences))

categ_id = dict(zip(list(set(categories)), np.arange(len(list(set(categories))))))
pd.DataFrame({'category': categ_id.keys(), 'id': categ_id.values()}).to_csv('categories.csv', index = False)

sent_df = pd.DataFrame({'title': titles, 'category': categories, 'sentence': all_sentences})
sent_df['categ_id'] = [categ_id[i] for i in sent_df.category.values]

print('Loading the model...')
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

print('Making embeddings...')
embeddings = model.encode(all_sentences)

print(embeddings.shape)

sent_df['sent_tr_emb'] = list(embeddings)
sent_df.to_csv('sentence_embeddings_500.csv', index = False)