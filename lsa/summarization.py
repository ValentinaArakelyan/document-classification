import numpy as np
from tqdm import tqdm
from lsa_summarizer import LsaSummarizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

source_file = "../../data/test_papers.txt"
output_file = '../../results/lsa_summaries.txt'

with open(source_file, "r", encoding='utf-8') as file:
    paper_texts = file.readlines()

summarizer = LsaSummarizer()

stopwords = stopwords.words('english')
summarizer.stop_words = stopwords

for paper in paper_texts:
    paper = paper.strip()
    paper_id = paper.split('\t')[0]
    keyword_count = int(paper.split('\t')[1])
    text = paper.split('\t')[2]
    
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    sent_per_topic = int(np.ceil(num_sentences / keyword_count / 3)) + 1
    
    concepts =summarizer(text, keyword_count, sent_per_topic)
    
    summary_sentences = []
    for concept in concepts:
        summary_sentences += concept
        
    summary_sentences = set(summary_sentences)
        
    print(paper_id, num_sentences, num_sentences // 3, len(summary_sentences))
    
    # Build summary by rearranging summary sentences as in original text
    summary = ''
    for orig_sent in sentences:
        if orig_sent in summary_sentences:
            summary += ' ' + orig_sent
    
    #print(summary)
    
    with open(output_file, 'a+', encoding='utf-8') as out_f:
        out_f.write(paper_id + '\t' + summary + '\n')

    #print(" ".join(summary))
