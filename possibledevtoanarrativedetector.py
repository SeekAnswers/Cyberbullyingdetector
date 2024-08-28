import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import docx2txt
import PyPDF2
import io

# Load the trained model and the TF-IDF vectorizer
model = pickle.load(open('LinearSVCTuned.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))

# Set up the Streamlit page
st.title('Cyberbullying Detection System')
st.write('Enter a text below or upload a file to check if it is bullying or not.')

# Option 1: Input text directly
user_input = st.text_area('Enter text here:', '')

# Option 2: Upload a file
uploaded_file = st.file_uploader('Or upload a file', type=['txt', 'csv', 'docx', 'pdf'])

# Initialize file content
file_content = ''

# Processing uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == 'txt':
        file_content = uploaded_file.read().decode('utf-8')
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file)
        file_content = '\n'.join(df.astype(str).values.flatten())
    elif file_type == 'docx':
        file_content = docx2txt.process(uploaded_file)
    elif file_type == 'pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        file_content = "\n".join([page.extract_text() for page in pdf_reader.pages])

    # Checkbox for displaying the uploaded file content
    show_file_content = st.checkbox('Show uploaded file content', value=False)
    if show_file_content:
        st.text_area('Uploaded File Content:', file_content, height=300)

# Split file content into lines
lines = file_content.splitlines()

# Combine user input and lines from the uploaded file
texts_to_predict = [user_input] if user_input else []
texts_to_predict.extend(lines)

if st.button('Predict'):
    if texts_to_predict:
        # Preprocess the input text(s) using the loaded TF-IDF vectorizer
        transformed_inputs = tfidf_vectorizer.transform(texts_to_predict)
        
        # Make predictions
        predictions = model.predict(transformed_inputs)
        
        # Display the results
        for i, text in enumerate(texts_to_predict):
            if predictions[i] == 1:
                st.error(f'Text {i+1}: **bullying** - "{text}"')
            else:
                st.success(f'Text {i+1}: **non-bullying** - "{text}"')
    else:
        st.warning('Please enter some text or upload a file to make a prediction.')
