from googletrans import Translator
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Translate
def translate(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def summarise(text):
    """
    NOTE:
    something went wrong while using cuda
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cuda')
        return summarizer(text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']
    except:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']