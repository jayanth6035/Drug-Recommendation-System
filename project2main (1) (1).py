#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import string

# Load the dataset
df = pd.read_csv('drugsCom_raw (1).csv', encoding='Latin-1')
df = df.drop(['Unnamed: 0', 'date', 'usefulCount'], axis=1)
df = df[(df['condition'] == 'Depression') | (df['condition'] == 'High Blood Pressure') | (df['condition'] == 'Diabetes, Type 2')]
df.dropna(axis=0, inplace=True)

# Define the conditions we want to focus on
CONDITIONS = {
    'Depression': ['Depression','antidepressant','mood swing'],
    'High Blood Pressure': ['High Blood Pressure', 'blood pressure','hypertension'],
    'Diabetes, Type 2': ['Diabetes, Type 2','diarrhea gas', 'diagnosed']
}

stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    # Delete HTML tags
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # Remove punctuation and convert to lowercase
    review_text = review_text.translate(str.maketrans('', '', string.punctuation))
    review_text = review_text.lower()

    # Remove numbers and other non-letter characters
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)

    # Remove stopwords and lemmatize words
    words = review_text.split()
    meaningful_words = [w for w in words if not w in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]

    # Return the cleaned review text
    return ' '.join(lemmatized_words)

df['review_clean'] = df['review'].apply(review_to_words)

def get_top_drugs(review_text, num=5):
    # Preprocess the review
    review_clean = review_to_words(review_text)

    # Filter the dataset based on the condition of the input review
    condition = None
    for key, values in CONDITIONS.items():
        for value in values:
            if value.lower() in review_clean.lower():
                condition = key
                break
        if condition:
            break
    if condition:
        df_condition = df[df['condition'] == condition]
    else:
        return []

    # Create a TF-IDF matrix for the reviews
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_condition['review_clean'])

    # Compute the cosine similarity matrix for the reviews
    cosine_sim = cosine_similarity(tfidf_matrix.T, tfidf_matrix.T)

    # Get the indices of the top recommended drugs
    indices = pd.Series(df_condition.index)
    indices_sorted = indices[cosine_sim[-1].argsort()[::-1]]
    top_indices = indices_sorted[1:num+1]

    # Get the top recommended drugs and their corresponding conditions
    top_drugs = df_condition.loc[top_indices, ['drugName', 'condition']].values.tolist()

    return top_drugs


# Define the Streamlit app
st.title('Drug Recommender')

# Get the review text from the user
review = st.text_input('Enter your review:')

# Show the input review
st.write('Input review:', review)

# Show the recommended drugs
if st.button('Get recommendations'):
    try:
        top_drugs = get_top_drugs(review)
        if len(top_drugs) > 0:
            st.write('Top recommended drugs:')
            for drug, condition in top_drugs:
                st.write(f'{drug} (for {condition})')
        else:
            st.write('No recommendations found for the input review.')
    except Exception as e:
        st.write('Error:', e)

