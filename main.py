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

# Function to create the dictionary and postings lists
def create_dictionary_and_postings(corpus_dir):
    dictionary = defaultdict(list)
    doc_lengths = {}
    doc_filenames = {}
    doc_id = 1
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            doc_filenames[doc_id] = filename
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                tokens = preprocess_text(text)
                doc_length = len(tokens)
                doc_lengths[doc_id] = doc_length
                for token in tokens:
                    dictionary[token].append((doc_id, tokens.count(token)))
                doc_id += 1
    return dictionary, doc_lengths, doc_filenames


# Function to calculate the tf-idf weights
def calculate_tf_idf_weights(dictionary, doc_lengths, N):
    tf_idf_weights = {}
    for term, postings in dictionary.items():
        df = len(postings)
        idf = math.log10(N / df)
        tf_idf_weights[term] = {}
        for doc_id, tf in postings:
            tf_idf_weights[term][doc_id] = (1 + math.log10(tf)) * idf
    return tf_idf_weights

# Function to search and rank documents
def search_and_rank(tf_idf_weights, doc_lengths, doc_filenames, query):
    query_tokens = preprocess_text(query)
    query_vector = {}
    for token in query_tokens:
        if token in tf_idf_weights:
            query_vector[token] = (1 + math.log10(query_tokens.count(token))) * math.log10(len(doc_lengths) / len(tf_idf_weights[token]))
    doc_scores = {}
    for doc_id in doc_lengths:
        doc_vector = {}
        for token, postings in tf_idf_weights.items():
            if doc_id in postings:
                doc_vector[token] = postings[doc_id]
        dot_product = sum(query_vector[token] * doc_vector[token] for token in query_vector if token in doc_vector)
        magnitude_query = math.sqrt(sum(query_vector[token] ** 2 for token in query_vector))
        magnitude_doc = math.sqrt(sum(doc_vector[token] ** 2 for token in doc_vector))
        cosine_similarity = dot_product / (magnitude_query * magnitude_doc)
        doc_scores[doc_id] = cosine_similarity
    return [(doc_filenames[doc_id], score) for doc_id, score in sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:10]]


def main():
    corpus_dir = 'corpus' 
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