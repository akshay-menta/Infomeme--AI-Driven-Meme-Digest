import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()

# Class for summarizing the content of a URL.
class URLContentSummarizer:
    def __init__(self):
        model_name = "Akshay-Sai/t5-small-xsum-finetuned-2epochs"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=model, tokenizer=self.tokenizer)

    # Extracting text from a URL.
    def extract_text_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text

        except Exception as e:
            raise Exception(f"Error extracting text from URL: {str(e)}")

    # Summarizing the text.
    def summarize_text(self, text, max_new_tokens=40, min_length=10):
        try:
            words = text.split()
            chunk_size = 400 
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            summaries = []
            for chunk in chunks:
                if len(chunk.strip().split()) < min_length:
                    continue
                summary = self.summarizer(chunk,
                                          max_new_tokens=max_new_tokens,
                                          min_length=min_length,
                                          truncation=True,
                                          no_repeat_ngram_size=3)
                summaries.append(summary[0]['summary_text'])

            return ' '.join(summaries)

        except Exception as e:
            raise Exception(f"Error summarizing text: {str(e)}")

    # Processing a URL.
    def process_url(self, url, max_new_tokens=40, min_length=10):
        try:
            extracted_text = self.extract_text_from_url(url)
            summary = self.summarize_text(extracted_text, max_new_tokens, min_length)
            return summary
        except Exception as e:
            raise Exception(f"Error processing URL: {str(e)}")
