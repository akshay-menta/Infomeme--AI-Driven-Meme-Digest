import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score, hamming_loss
from tqdm import tqdm
import datasets
from collections import Counter
import Levenshtein

# Selecting GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Setting seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Loading the dataset
def load_dataset_goemotions():
    return datasets.load_dataset("go_emotions")

# Converting dataset to DataFrames, remove neutral-only samples, and return label counts
def process_goemotions(dataset):
    labels = dataset["train"].features["labels"].feature.names
    neutral_index = labels.index("neutral")

    def filter_neutral_only(df):
        return df[df["labels"].apply(lambda x: neutral_index not in x or len(x) > 1)].reset_index(drop=True)

    train_df = filter_neutral_only(pd.DataFrame(dataset["train"]))
    val_df = filter_neutral_only(pd.DataFrame(dataset["validation"]))
    test_df = filter_neutral_only(pd.DataFrame(dataset["test"]))

    counts = Counter()
    for row in train_df["labels"]:
        for label in row:
            counts[labels[label]] += 1

    return train_df, val_df, test_df, labels, counts

# Augmenting the text by simple random swaps or deletions
def augment_text(text):
    words = text.split()
    if len(words) > 3 and random.random() > 0.5:
        i1, i2 = random.sample(range(len(words)), 2)
        words[i1], words[i2] = words[i2], words[i1]
    if len(words) > 3 and random.random() > 0.7:
        words.pop(random.randint(0, len(words) - 1))
    return " ".join(words)

# Balancing the dataset by augmenting underrepresented emotion classes
def balance_dataset(train_df, emotion_labels, label_counts):
    max_count = max(label_counts.values())
    target = {label: int(max_count * 0.5) for label in label_counts}
    current = {label: 0 for label in label_counts}
    texts, labels = [], []
    for _, row in train_df.iterrows():
        text, lbls = row["text"], row["labels"]
        if not lbls:
            continue
        for l in lbls:
            name = emotion_labels[l]
            current[name] += 1
            if current[name] < target[name]:
                for _ in range(2):
                    texts.append(augment_text(text))
                    labels.append(lbls)
    return pd.concat([train_df, pd.DataFrame({"text": texts, "labels": labels})], ignore_index=True)

# Creating the dataset class
class GoEmotionsT5Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, emotion_labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.emotion_labels = emotion_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = f"Identify the emotions expressed in this text: {self.texts[idx]}"
        label_names = [self.emotion_labels[i] for i in self.labels[idx]]
        target_text = ", ".join(label_names) if label_names else "neutral"

        input_enc = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        target_enc = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }

# Function to predict the labels
def predict(text, model, tokenizer, emotion_labels, device):
    model.eval()
    input_text = f"Identify the emotions expressed in this text: {text}"
    tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        output = model.generate(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], max_length=64)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted_emotions = []
    for part in decoded.split(","):
        word = part.strip().lower()
        for label in emotion_labels:
            if Levenshtein.distance(word, label.lower()) <= 2:
                predicted_emotions.append(label)
    return list(set(predicted_emotions)) or ["neutral"]

# Evaluating the model
def evaluate(model, dataloader, tokenizer, emotion_labels):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            out = model.generate(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device),
                                 max_length=64)
            pred_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for pred, true in zip(pred_texts, target_texts):
                pred_labels = [label for label in emotion_labels if any(Levenshtein.distance(p.strip(), label.lower()) <= 2 for p in pred.split(","))]
                true_labels = [t.strip() for t in true.split(",")]
                pred_vec = [1 if label in pred_labels else 0 for label in emotion_labels]
                true_vec = [1 if label in true_labels else 0 for label in emotion_labels]
                preds.append(pred_vec)
                targets.append(true_vec)

    print("Micro F1:", f1_score(targets, preds, average="micro", zero_division=0))
    print("Macro F1:", f1_score(targets, preds, average="macro", zero_division=0))
    print("Hamming Accuracy:", 1 - hamming_loss(targets, preds))
    print("Jaccard Score:", jaccard_score(targets, preds, average="samples", zero_division=0))

# Model training function
def train(model, tokenizer, train_dl, val_dl, emotion_labels):
    optimizer = AdamW(model.parameters(), lr=5e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * len(train_dl) * 3,
                                                num_training_steps=len(train_dl) * 3)
    model.to(device)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += output.loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dl):.4f}")
        evaluate(model, val_dl, tokenizer, emotion_labels)

# Main function
if __name__ == "__main__":
    set_seed()

    dataset = load_dataset_goemotions()
    train_df, val_df, test_df, emotion_labels, label_counts = process_goemotions(dataset)
    train_df = balance_dataset(train_df, emotion_labels, label_counts)

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    train_set = GoEmotionsT5Dataset(train_df["text"], train_df["labels"], tokenizer, emotion_labels)
    val_set = GoEmotionsT5Dataset(val_df["text"], val_df["labels"], tokenizer, emotion_labels)
    test_set = GoEmotionsT5Dataset(test_df["text"], test_df["labels"], tokenizer, emotion_labels)

    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=16)
    test_dl = DataLoader(test_set, batch_size=16)

    train(model, tokenizer, train_dl, val_dl, emotion_labels)

    save_path = "t5_goemotions_finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    sample_inputs = [
        "I'm feeling really sad and down today.",
        "Wow, that just made me so happy!",
        "I'm not sure what to feel about this situation.",
        "You did an amazing job, I'm proud of you.",
        "This is annoying and I'm getting frustrated.",
        "I hope this works and this is my last chance.",
        "I have no idea.",
        "We need to push really hard."
    ]

    for text in sample_inputs:
        emotions = predict(text, model, tokenizer, emotion_labels, device)
        print(f"Text: {text}")
        print(f"Predicted emotions: {emotions}\n")
