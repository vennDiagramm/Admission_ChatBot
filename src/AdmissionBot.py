import PyPDF2
import pandas as pd
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's language model
nlp = spacy.load('en_core_web_sm')

def process_text(text):
    doc = nlp(text)
    return doc

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


# Function to compute the chatbot response based on file content
def chatbot_response(question, text_data):
    sentences = [sent.text for sent in nlp(text_data).sents] # Split text into sentences using spaCy
    
    # Use TF-IDF to calculate similarity between the question and sentences in the text
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    vectors = vectorizer.toarray()
    
    # Compute similarity between the question and each sentence
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Find the sentence with the highest similarity score
    best_match_idx = cosine_similarities.argmax()
    best_answer = sentences[best_match_idx]
    
    return best_answer


# Streamlit interface
def main():
    st.title("File-Based Chatbot")
    st.write("Upload a PDF or CSV file, and ask questions based on the file content.")
    
    uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'csv'])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text_data = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/csv":
            text_data = extract_data_from_csv(uploaded_file)
        st.write("File Content:", text_data[:500], "...")
    
    question = st.text_input("Ask a question based on the file content:")
    
    if question:
        response = chatbot_response(question, text_data)
        st.write("Response:", response)

if __name__ == '__main__':
    main()