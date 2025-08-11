import os
import warnings
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import classification_report, confusion_matrix
import evaluate

warnings.filterwarnings("ignore")

# Configuration
Model_name = "roberta-base"
Num_labels = 6
Max_length = 256
Seed = 42
Output_dir = "./liar2-roberta-base-finetuned"
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading LIAR2 dataset and creating a combined text field.
def load_and_prepare_dataset():
    dataset = load_dataset("chengxuphd/liar2")

    def add_combined_text(example):
        return {
            "text": (
                f"[SPEAKER] {example['speaker']} "
                f"[SUBJECT] {example['subject']} "
                f"[CONTEXT] {example['context']} "
                f"[STATEMENT] {example['statement']} "
                f"[JUSTIFICATION] {example['justification']}"
            )
        }

    dataset = dataset.map(add_combined_text)
    dataset = dataset.rename_column("label", "labels")
    return dataset

# Tokenizing combined text field for RoBERTa.
def tokenize_dataset(dataset, tokenizer):
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=Max_length
        )

    columns_to_remove = [col for col in dataset["train"].column_names if col != "labels"]
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

# Computing accuracy, macro F1, and weighted F1.
def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"],
    }

# Training model with early stopping and return Trainer.
def train_model(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=Output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.1,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=f"{Output_dir}/logs",
        logging_steps=100,
        seed=Seed,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(Output_dir)
    tokenizer.save_pretrained(Output_dir)
    return trainer

# Evaluating model on test set and save results.
def evaluate_model(trainer, tokenized_dataset):
    predictions = trainer.predict(tokenized_dataset["test"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    label_names = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    report = classification_report(y_true, y_pred, target_names=label_names)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)

    with open(os.path.join(Output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(Output_dir, "test_confusion_matrix.txt"), "w") as f:
        np.savetxt(f, conf_matrix, fmt="%d")

# Predicting label and probabilities for input text.
def predict_fake_news(text, model, tokenizer, device=Device):
    model.eval()
    model.to(device)

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    label_names = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

    return {
        "prediction": label_names[predicted_class],
        "confidence": confidence,
        "all_probabilities": {label_names[i]: predictions[0][i].item() for i in range(Num_labels)}
    }

# Main function
def main():
    set_seed(Seed)

    dataset = load_and_prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(Model_name)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(Model_name, num_labels=Num_labels)

    trainer = train_model(model, tokenizer, tokenized_dataset)
    evaluate_model(trainer, tokenized_dataset)

    sample_texts = [
        "The Earth is flat and has been proven by scientists worldwide.",
        "COVID-19 vaccines have been shown to be safe and effective in clinical trials.",
        "The 2020 US presidential election was free and fair according to election officials.",
        "India Becomes First Country to Land Near Moon’s South Pole.",
        "Northeastern university main campus is located in Boston.",
        "Northeastern university main campus is located in India.",
        "Nasa has found out about aliens.",
        "Drinking a pink salt diet beverage will result in significant weight loss.",
        "A CDC study found that the majority of those infected with COVID-19 'always' wore masks.",
        "Project 2025 proposes women should be forced to carry 'period passports' to track their menstrual cycles."
    ]

    print("\nSample Predictions:")
    for i, text in enumerate(sample_texts, 1):
        result = predict_fake_news(text, model, tokenizer)
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        top3 = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for label, prob in top3:
            print(f"  {label}: {prob:.3f}")

if __name__ == "__main__":
    main()
