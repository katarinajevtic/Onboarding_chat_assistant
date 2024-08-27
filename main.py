import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

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


# NLP Model Training
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
