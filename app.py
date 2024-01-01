# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# from flask import Flask, request, jsonify
# import pandas as pd
# import tensorflow_hub as hub
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# # Load Universal Sentence Encoder
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Function to calculate USE embeddings
# def get_use_embeddings(texts):
#     embeddings = embed(texts)
#     return np.array(embeddings)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     text1 = data['text1']
#     text2 = data['text2']

#     embeddings_row1 = get_use_embeddings([text1])
#     embeddings_row2 = get_use_embeddings([text2])

#     similarity = float(cosine_similarity(embeddings_row1, embeddings_row2)[0][0])

#     result = {'text1': text1, 'text2': text2, 'cosine_similarity': similarity}
#     return jsonify(result) 

# if __name__ == '__main__':
#     app.run(port=5000)



from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to calculate USE embeddings
def get_use_embeddings(texts):
    embeddings = embed(texts)
    return np.array(embeddings)

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

#if __name__ == '__main__':
#    app.run(port=5000)
