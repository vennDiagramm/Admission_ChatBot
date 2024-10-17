import PyPDF2
import pandas as pd
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# this is for text extracting - PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfFileReader(pdf_file)
    text = ''
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extract_text()
    return text

# text extraction - csv
def extract_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_string()  # Convert dataframe to a string for easier handling

### FOR NLP TASK ###

# Load spaCy's language model
nlp = spacy.load('en_core_web_sm')

def process_text(text):
    doc = nlp(text)
    return doc

# Function to compute the chatbot response based on file content
def chatbot_response(question, text_data):
    sentences = [sent.text for sent in nlp(text_data).sents]
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_idx = cosine_similarities.argmax()
    best_answer = sentences[best_match_idx]
    return best_answer