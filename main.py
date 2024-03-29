from flask import Flask, render_template, request
from transformers import pipeline
from spacy.matcher import Matcher
import spacy
import os

app = Flask(__name__)

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize matcher
matcher = Matcher(nlp.vocab)

# Define patterns for entities and noun phrases
patterns = [
    [{"POS": "NOUN"}, {"POS": "NOUN"}],  # Noun phrases
    [{"ENT_TYPE": "PERSON"}, {"ENT_TYPE": "ORG"}],  # Entities
]

matcher.add("CLAUSE_PATTERNS", patterns)

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Check if the file is a text file
        if file and file.filename.endswith('.txt'):
            # Save the uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Process the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Process a document using SpaCy
            doc = nlp(text)

            # Find matches using SpaCy matcher
            matches = matcher(doc)
            matched_texts = [doc[start:end].text for _, start, end in matches]

            # Extracted text for summarization
            extracted_text = text

            # Adjust max_length based on the length of extracted_text
            max_length = min(150, len(extracted_text.split()))

            # Summarize the text
            summary = summarizer(extracted_text, max_length=max_length, min_length=30, do_sample=False)
            summary_text = summary[0]['summary_text']

            return render_template('index.html', text=text, matched_texts=matched_texts, summary=summary_text)
        else:
            return render_template('index.html', error='Please upload a .txt file')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
