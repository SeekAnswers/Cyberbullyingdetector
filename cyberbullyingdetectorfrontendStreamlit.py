import streamlit as st
import pickle
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and the fitted TF-IDF vectorizer
model = pickle.load(open('LinearSVCTuned.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))

# Streamlit UI setup
st.title('Seek Answers Cyberbullying Detection System')
st.write('Enter text below or upload a file to check if it is bullying or not.')

# Option 1: Input text directly
user_input = st.text_area('Enter text here:', '')

# Option 2: Upload a file
uploaded_file = st.file_uploader('Or upload a file', type=['txt', 'docx', 'pdf'])

# Initialize text container
file_content = ""

# Processing uploaded file
if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        file_content = uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.docx'):
        file_content = docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        file_content = "\n".join([page.extract_text() for page in reader.pages])

    # Checkbox to show/hide file content
    show_file_content = st.checkbox('Show uploaded file content', value=False)
    
    if show_file_content:
        # Display file content in a scrollable text area
        st.text_area('Uploaded File Content:', file_content, height=300)

# Combine user input and lines from the uploaded file
texts_to_predict = [user_input] if user_input else []
texts_to_predict.extend(file_content.splitlines())

# Prediction
if st.button('Predict'):
    if texts_to_predict:
        # Preprocess the input text(s)
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
