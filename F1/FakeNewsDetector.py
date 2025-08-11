import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', message='`encoder_attention_mask` is deprecated')

# Class for detecting fake news.
class FakeNewsDetector:
    def __init__(self, model_name: str = "SaiRakshith/liar2-roberta-base-finetuned"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading fake news detection model from: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_names = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

    # Removing source names and other common suffixes.
    @staticmethod
    def remove_source_names(text: str) -> str:
        source_patterns = [
            r'\s*[-–]\s*[A-Za-z0-9\s\.]+\.[A-Za-z]+',  
            r'\s*[-–]\s*[A-Za-z0-9\s\.]*(News|Times|Post|Daily|Tribune|Journal|Press|Media|Network|Deadline|Axios|NPR)',  
            r'\s*[-–]\s*[A-Za-z0-9\s\.]*\s+(News|Times|Post|Daily|Tribune|Journal|Press|Media|Network)',  
            r'\s*[-–]\s*(Reuters|AP|AFP|Bloomberg|NBC|ABC|CBS|CNN|BBC|Fox|CNBC|NPR|Axios|Deadline)', 
            r'\s*[-–]\s*The\s+[A-Za-z\s]+', 
        ]
        cleaned_text = text
        for pattern in source_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\s*[-–]\s*$', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

    # Preprocessing the text.
    @staticmethod
    def preprocess_text(text: str) -> str:
        cleaned = FakeNewsDetector.remove_source_names(text)
        cleaned = cleaned.replace(" - ", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    # Predicting the fake news label.
    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_idx].item()

        return {
            'prediction': self.label_names[predicted_class_idx],
            'confidence': confidence,
            'all_probabilities': {
                self.label_names[i]: probabilities[0][i].item() 
                for i in range(len(self.label_names))
            }
        }

    # Predicting the fake news label for a list of news articles with confidence threshold of 0.7.
    def predict_batch(self, news_list: List[str], confidence_threshold: float = 0.7,progress: bool = True) -> List[Dict]:
        results = []
        iterable = tqdm(news_list, desc="Predicting", disable=not progress)

        for text in iterable:
            clean_text = self.preprocess_text(text)
            pred = self.predict(clean_text)

            final_label = "true" if (
                pred['prediction'] == 'true' and 
                pred['confidence'] >= confidence_threshold
            ) else pred['prediction']

            results.append({
                'original_text': text,
                'clean_text': clean_text,
                'predicted_label': pred['prediction'],
                'confidence': pred['confidence'],
                'final_label': final_label,
                'all_probabilities': pred['all_probabilities']
            })

        return results

    # Filtering and returning only the true news articles.
    @staticmethod
    def filter_true_news(n,prediction_results: List[Dict]) -> List[str]:
        return [item['clean_text'] for item in prediction_results if item['final_label'] == 'true'][:n]
