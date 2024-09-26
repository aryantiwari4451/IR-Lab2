# IR LAB 2 Asignment

import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# pre processing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens







def main():
    corpus_dir = 'corpus'  # replace with your corpus directory
    dictionary, doc_lengths, doc_filenames = create_dictionary_and_postings(corpus_dir)
    N = len(doc_lengths)
    tf_idf_weights = calculate_tf_idf_weights(dictionary, doc_lengths, N)
    query = input("Enter your query: ")
    ranked_docs = search_and_rank(tf_idf_weights, doc_lengths, doc_filenames, query)
    for doc in ranked_docs:
        print(doc)
        
if __name__ == "__main__":
    import math
    main()