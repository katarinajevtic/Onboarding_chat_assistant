import nltk
import os

# Define the custom NLTK data directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add the custom directory to NLTK's data paths
nltk.data.path.append(nltk_data_path)

# Download required NLTK datasets
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Rest of your imports and code
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load intents and entities
with open('intents.json', 'r') as file:
    intents = json.load(file)

with open('entities.json', 'r') as file:
    entities = json.load(file)

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Train NLP model
X = []
y = []
for intent, data in intents.items():
    for question in data['questions']:
        X.append(preprocess(question))
        y.append(intent)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_train_counts, y)

# Dialogue management
def chatbot_response(user_input):
    user_input_preprocessed = preprocess(user_input)
    user_input_vector = vectorizer.transform([user_input_preprocessed])
    intent = model.predict(user_input_vector)[0]

    if intent in intents:
        questions = intents[intent]["questions"]
        answers = intents[intent]["answers"]

        question_vectors = vectorizer.transform([preprocess(q) for q in questions])

        similarities = cosine_similarity(user_input_vector, question_vectors)
        most_similar_index = np.argmax(similarities)

        if similarities[0][most_similar_index] > 0.7:  # Adjust similarity threshold as needed
            return answers[most_similar_index]
        else:
            return "I'm sorry, I'm not sure how to help with that."
    else:
        return "I'm sorry, I'm not sure how to help with that."

# Streamlit UI
st.image("logo.png", width=200)
st.title("Onboarding Assistant")
st.write("Welcome to the Onboarding Assistant Chatbot! How can I help you today?")

user_input = st.text_input("You: ", "")
if user_input:
    bot_response = chatbot_response(user_input)
    st.text_area("Assistant Chatbot:", value=bot_response, height=100)

# Dialogue Management
def chatbot_response(user_input):
    user_input_preprocessed = preprocess(user_input)
    user_input_vector = vectorizer.transform([user_input_preprocessed])
    intent = model.predict(user_input_vector)[0]

    if intent in intents:
        questions = intents[intent]["questions"]
        answers = intents[intent]["answers"]

        question_vectors = vectorizer.transform([preprocess(q) for q in questions])

        similarities = cosine_similarity(user_input_vector, question_vectors)
        most_similar_index = np.argmax(similarities)

        if similarities[0][most_similar_index] > 0.7:  # Adjusting similarity threshold as needed
            return answers[most_similar_index]
        else:
            return "I'm sorry, I'm not sure how to help with that."
    else:
        return "I'm sorry, I'm not sure how to help with that."


# Add image
st.image("logo.png", width=200)
# Streamlit UI
st.title("Onboarding Assistant")
st.write("Welcome to the Onboarding Assistant Chatbot! How can I help you today?")

user_input = st.text_input("You: ", "")
if user_input:
    bot_response = chatbot_response(user_input)
    st.text_area("Assistant Chatbot:", value=bot_response, height=100)
