import nltk
import numpy as np
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

corpus = """
Hello! I am your chatbot assistant.
How can I help you today?
I can answer questions about weather, time, or general queries.
The weather today is sunny and warm.
The current time is 5:00 PM.
I am doing well, thank you!
What is your name?
I am a chatbot created using Python.
Goodbye! Have a nice day.
"""

sent_tokens = nltk.sent_tokenize(corpus)

GREETING_INPUTS = ("hello", "hi", "greetings", "hey", "what's up")
GREETING_RESPONSES = ["Hi", "Hey", "Hello!", "I am glad to talk to you!"]

def greet(sentence):
    for word in sentence.lower().split():
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def generate_response(user_input):
    sent_tokens.append(user_input)
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    if score == 0:
        response = "Sorry, I didn't understand that."
    else:
        response = sent_tokens[idx]

    sent_tokens.pop()
    return response

print("ChatBot: Hello! Ask me anything or type 'bye' to exit.")

while True:
    user_input = input("You: ")
    user_input = user_input.lower()

    if user_input in ['bye', 'exit', 'quit']:
        print("ChatBot: Goodbye!")
        break
    elif greet(user_input):
        print("ChatBot:", greet(user_input))
    else:
        print("ChatBot:", generate_response(user_input))
