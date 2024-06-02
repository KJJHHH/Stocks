from googletrans import Translator
from transformers import pipeline
from huggingface_hub.hf_api import HfFolder
import json, pickle
import warnings
warnings.filterwarnings("ignore")
HfFolder.save_token('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')

# ------------
# STORE and LOAD
encoding = 'cp950'
def store_json(file_path, data):
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f) 
        
def load_json(file_path):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)

def store_pickle(file_path, data):
    with open(file_path, 'ab') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Translate
def translate(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def summarise(text):
    pass
    return text
    """
    NOTE:
    something went wrong while using cuda
    """
    """
    import transformers
    import torch

    model_id = "meta-llama/Meta-Llama-3-8B"

    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )
    print(pipeline("Hey how are you doing today?"))
    ---
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from transformers.utils.hub import move_cache
    
    move_cache()
    model_id = "meta-llama/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    """

    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=30, min_length=30, do_sample=True)[0]['summary_text']
    """
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    
    return (tokenizer.decode(outputs[0]))
    """

    
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cuda')
        return summarizer(text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']
    except:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']
    
    """
        