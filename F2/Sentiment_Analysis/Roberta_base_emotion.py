import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, jaccard_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Selecting GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Setting seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Loading the dataset
def load_goemotions():
    return load_dataset("go_emotions")

# Removing neutral labels
def remove_neutral(df, labels):
    if 'neutral' not in labels:
        return df
    idx = labels.index('neutral')
    df = df[~df['labels'].apply(lambda x: idx in x)]
    return df.reset_index(drop=True)

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
def balance_dataset(df, labels):
    counts = {label: 0 for label in labels}
    for label_list in df['labels']:
        for l in label_list:
            counts[labels[l]] += 1
    max_count = max(counts.values())
    target = {label: int(max_count * 0.5) for label in labels}
    current = {label: 0 for label in labels}
    new_texts, new_labels = [], []

    for _, row in df.iterrows():
        for l in row["labels"]:
            name = labels[l]
            current[name] += 1
            if current[name] < target[name]:
                for _ in range(2):
                    new_texts.append(augment_text(row["text"]))
                    new_labels.append(row["labels"])

    aug_df = pd.DataFrame({"text": new_texts, "labels": new_labels})
    return pd.concat([df, aug_df], ignore_index=True)

# Processing the dataset
def process_dataset(dataset):
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    labels = dataset['train'].features['labels'].feature.names
    train_df = remove_neutral(train_df, labels)
    val_df = remove_neutral(val_df, labels)
    test_df = remove_neutral(test_df, labels)
    return train_df, val_df, test_df, labels, len(labels)

# Creating the dataset class
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# Evaluating the model with different metrics
def evaluate(true_labels, preds, mode="Validation"):
    micro_f1 = f1_score(true_labels, preds, average='micro')
    macro_f1 = f1_score(true_labels, preds, average='macro')
    precision = precision_score(true_labels, preds, average='micro')
    recall = recall_score(true_labels, preds, average='micro')
    hamming = hamming_loss(true_labels, preds)
    jaccard = jaccard_score(true_labels, preds, average='micro')
    print(f"\n[{mode}] Micro-F1: {micro_f1:.4f} Macro-F1: {macro_f1:.4f} "
          f"Precision: {precision:.4f} Recall: {recall:.4f} "
          f"Hamming: {hamming:.4f} Jaccard: {jaccard:.4f}")

# Model training function
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = (torch.sigmoid(outputs.logits).cpu().numpy() >= 0.5).astype(int)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")
        evaluate(all_labels, all_preds, mode="Validation")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'roberta_goemotions_best.pt')

# Model testing
def test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (torch.sigmoid(outputs.logits).cpu().numpy() >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    evaluate(all_labels, all_preds, mode="Test")

# Function to predict the labels
def predict(texts, model_path="roberta_goemotions_best.pt", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("go_emotions")
    labels = dataset['train'].features['labels'].feature.names
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(labels), problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits).cpu().numpy()[0]
        pred_labels = [labels[i] for i, p in enumerate(probs) if p >= threshold]
        results.append(pred_labels)
    return results

# Main function
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_goemotions()
    train_df, val_df, test_df, labels, num_labels = process_dataset(dataset)
    train_df = balance_dataset(train_df, labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    to_one_hot = lambda lbls: [1 if i in lbls else 0 for i in range(num_labels)]
    train_data = GoEmotionsDataset(train_df['text'].tolist(), [to_one_hot(l) for l in train_df['labels']], tokenizer)
    val_data = GoEmotionsDataset(val_df['text'].tolist(), [to_one_hot(l) for l in val_df['labels']], tokenizer)
    test_data = GoEmotionsDataset(test_df['text'].tolist(), [to_one_hot(l) for l in test_df['labels']], tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=num_labels, problem_type="multi_label_classification"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * 3)

    train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3)
    model.load_state_dict(torch.load('roberta_goemotions_best.pt'))
    test(model, test_loader, device)

    save_path = "roberta_goemotions_finetuned"
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
        emotions = predict(text, model, tokenizer, labels, device)
        print(f"Text: {text}")
        print(f"Predicted emotions: {emotions}\n")

if __name__ == "__main__":
    main()
