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


# interface
def main():
    st.set_page_config(page_title="Megatron",page_icon="🤖") # pulihan ug mcm 
    st.header("Megatron, Admission Buddy :books:")
    st.write("This chatbot answers questions based on the content of a PDF or CSV file in the specified directory.")
    
    # Specify the file path
    file_path = 'path/to/your/file.pdf'  # You can change this to the path of your PDF or CSV file
    
    # Determine file type based on extension
    if file_path.endswith(".pdf"):
        text_data = extract_text_from_pdf(file_path)
    elif file_path.endswith(".csv"):
        text_data = extract_data_from_csv(file_path)
    else:
        st.error("Unsupported file format. Please use PDF or CSV.")
        return
    
    # Show a snippet of the content (optional)
    st.write("File Content Preview:", text_data[:500], "...")
    
    # User question input
    question = st.text_input("Ask a question based on the file content:")
    
    if question:
        response = chatbot_response(question, text_data)
        st.write("Response:", response)

if __name__ == '__main__':
    main()