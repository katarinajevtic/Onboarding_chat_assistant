# Onboarding Chat Assistant
Onboarding Chat Assistant is a simple chatbot designed to assist with onboarding tasks, utilizing Natural Language Processing (NLP) to understand and respond to user queries.
## Features
+ NLP-based: The chatbot uses NLTK for text preprocessing and Scikit-learn's Naive Bayes classifier for intent recognition.
+ Customizable: You can easily extend the chatbot by modifying the `intents.json` and ` entities.json `files.
+ Similarity Matching: Uses cosine similarity to match user queries with predefined questions to provide the most relevant response.
Streamlit UI: The chatbot features a user-friendly interface built with Streamlit.

## Project Structure
+ intents.json: Contains the intents, including questions and answers.
+ entities.json: Defines the entities recognized by the chatbot.
+ main.py: The main script that powers the chatbot.
+ logo.png: The logo displayed in the Streamlit UI.

## Installation
To run the Onboarding Chat Assistant, you'll need to install the necessary dependencies:

```
pip install nltk scikit-learn streamlit
```

Download the NLTK data files:

```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
## Usage
1. Clone the repository:
2. Run the Streamlit application:
   ```
   streamlit run main.py
   ```
 3. Interact with the chatbot:

+ Enter your query in the text input field.
+ The chatbot will process your input and respond with the most relevant answer.

  ## 

![hello_onboarding](https://github.com/user-attachments/assets/cc74e9a7-70be-4499-a036-eb3b18142e51)


![onboarding_assistant1](https://github.com/user-attachments/assets/d2f379f3-bb81-4bd1-b1c6-caf8b02ba395)

   
