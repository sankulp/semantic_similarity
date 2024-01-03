import streamlit as st
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import string
import spacy

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nlp = spacy.load("en_core_web_sm")

# Function to calculate USE embeddings
def get_use_embeddings(texts):
    processed_texts = []
    for sentence in texts:
        doc = nlp(''.join([char.lower() for char in sentence if char not in string.punctuation]))
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        processed_texts.append(lemmatized_text)
    embeddings = embed(processed_texts)
    return np.array(embeddings)

def main():
    st.title("Cosine Similarity Calculator")

    # Input text fields
    text1 = st.text_area("Enter Text 1:", "")
    text2 = st.text_area("Enter Text 2:", "")

    if st.button("Calculate Cosine Similarity"):
        # Check if both text fields are not empty
        if text1 and text2:
            # Get USE embeddings
            embeddings_row1 = get_use_embeddings([text1])
            embeddings_row2 = get_use_embeddings([text2])

            # Calculate cosine similarity
            similarity = cosine_similarity(embeddings_row1, embeddings_row2)[0][0]

            # Display the result
            st.success(f"Cosine Similarity: {similarity:.4f}")
        else:
            st.warning("Please enter text in both fields.")

if __name__ == "__main__":
    main()