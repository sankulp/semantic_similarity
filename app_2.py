from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow_hub as hub
import numpy as np
import streamlit as st 
from sklearn.metrics.pairwise import cosine_similarity
import string 
import spacy


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nlp = spacy.load("en_core_web_sm")

# Function to calculate USE embeddings
# def get_use_embeddings(texts):
#     texts = [''.join([char.lower() for char in sentence if char not in string.punctuation]) for sentence in texts]
#     embeddings = embed(texts)
#     return np.array(embeddings)

def get_use_embeddings(texts):
    # Remove punctuation, convert to lowercase, and perform lemmatization
    processed_texts = []
    for sentence in texts:
        doc = nlp(''.join([char.lower() for char in sentence if char not in string.punctuation]))
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        processed_texts.append(lemmatized_text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text1 = data['text1']
    text2 = data['text2']

    embeddings_row1 = get_use_embeddings([text1])
    embeddings_row2 = get_use_embeddings([text2])

    similarity = float(cosine_similarity(embeddings_row1, embeddings_row2)[0][0])

    result = {'text1': text1, 'text2': text2, 'cosine_similarity': similarity}
    return jsonify(result) 

if __name__ == '__main__':
    app.run(debug = False , host = '0.0.0.0')

