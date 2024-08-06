# models/summarization_model.py

from transformers import pipeline
from text_extraction_model import extract_text

# Load pre-trained summarization model
summarizer = pipeline("summarization")

def summarize_attributes(text_data):
    summaries = {}
    for object_id, text in text_data.items():
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        summaries[object_id] = summary[0]['summary_text']

    return summaries

# Test the function
if __name__ == "__main__": 
    text_data = extract_text()
    summarize_attributes(text_data)
