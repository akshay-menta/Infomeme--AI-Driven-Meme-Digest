import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

# Class for analyzing emotions in news.
class EmotionAnalyzer:
    def __init__(self, model_name: str = "Akshay-Sai/goemotions-roberta-noneutral"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading emotion analysis model from: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # GoEmotions dataset labels
        self.labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude",
            "grief", "joy", "love", "nervousness", "optimism", "pride",
            "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]

    # Predicting the dominant emotion for each text in the list.
    def predict_emotion(self, text_list: List[str]) -> List[Dict]:
        results = []

        for text in text_list:
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
                probs = torch.sigmoid(logits)[0] 

            probs_dict = {self.labels[i]: probs[i].item() for i in range(len(self.labels))}
            top_label = max(probs_dict.items(), key=lambda x: x[1])[0]

            results.append({
                "text": text,
                "emotion": top_label,
                "probabilities": probs_dict
            })

        return results

    # Preparing simplified data structure with just news and primary emotion
    def prepare_for_meme_generation(emotion_results: List[Dict]) -> List[Dict]:
        simplified_data = []

        for result in emotion_results:
            simplified_item = {
                'news': result['text'],
                'sentiment': result['emotion']
            }
            simplified_data.append(simplified_item)

        return simplified_data

