import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from evaluate import load

# Configuration
Model_name = "t5-small"
Dataset_name = "xsum"
Output_dir = "./t5-small-xsum-finetuned"
Max_input_length = 512
Max_target_length = 64

# Main function: Orchestrates model loading, data preparation, training, and evaluation.
def run_training():
    # Disable W&B logging
    os.environ["WANDB_DISABLED"] = "true"

    # Device and Model Setup
    print("="*60)
    print("1. Setting up device and model")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model and tokenizer for '{Model_name}'...")
    tokenizer = T5Tokenizer.from_pretrained(Model_name)
    model = T5ForConditionalGeneration.from_pretrained(Model_name)
    model.to(device)

    # Dataset Analysis (Optional)
    if False:
        explore_and_analyze_dataset(tokenizer)

    # Load and Tokenize Data
    print("\n" + "="*60)
    print("2. Loading and preparing data")
    print("="*60)

    dataset = load_dataset(Dataset_name, trust_remote_code=True)

    # Tokenize documents and summaries for T5 summarization format.
    def tokenize_function(examples):
        inputs = ["summarize: " + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=Max_input_length, truncation=True, padding=False)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=Max_target_length, truncation=True, padding=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset"
    )

    print(f"Training samples: {len(tokenized_datasets['train']):,}")
    print(f"Validation samples: {len(tokenized_datasets['validation']):,}")
    print(f"Test samples: {len(tokenized_datasets['test']):,}")

    # Trainer Configuration
    print("\n" + "="*60)
    print("3. Configuring the trainer")
    print("="*60)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
    rouge_metric = load("rouge")

    # Compute ROUGE metrics for evaluation.
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=Output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge2",
        greater_is_better=True,
        fp16=True,
        predict_with_generate=True,
        logging_dir=f"{Output_dir}/logs",
        logging_steps=100,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    print("\n" + "="*60)
    print("4. Starting training")
    print("="*60)
    trainer.train()

    # Save and Evaluate
    print("\n" + "="*60)
    print("5. Saving and evaluating final model")
    print("="*60)

    trainer.save_model(Output_dir)
    print("\nEvaluating on the test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])

    print("\nTest Set ROUGE Scores:")
    for key, value in test_results.items():
        if 'rouge' in key or 'gen_len' in key:
            print(f"  {key}: {value:.4f}")

    # Custom Sample Test
    print("\n" + "="*60)
    print("6. Testing with a custom sample")
    print("="*60)

    custom_text = (
        "The James Webb Space Telescope, a collaboration between NASA, ESA, and CSA, launched in December 2021. "
        "It is the largest and most powerful space telescope ever built, designed to see the universe's first "
        "galaxies and stars. Its primary mirror is composed of 18 hexagonal segments made of gold-plated beryllium. "
        "The telescope operates in a halo orbit around the second Lagrange point (L2), nearly a million miles from "
        "Earth, to keep it cold and free from interference."
    )

    input_text = "summarize: " + custom_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=Max_input_length, truncation=True).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=Max_target_length,
            min_length=10,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"\nOriginal text:\n{custom_text}")
    print(f"\nGenerated summary:\n{generated_summary}")


# Helper function: To inspect dataset sizes and token length distributions
def explore_and_analyze_dataset(tokenizer):
    print("\n" + "="*80)
    print("Exploring and analyzing dataset")
    print("="*80)

    dataset = load_dataset(Dataset_name, trust_remote_code=True)
    print(f"- Train samples: {len(dataset['train']):,}")
    print(f"- Validation samples: {len(dataset['validation']):,}")
    print(f"- Test samples: {len(dataset['test']):,}")

    sample_size = 5000
    train_sample = dataset['train'].select(range(min(sample_size, len(dataset['train']))))

    doc_lengths = [len(tokenizer("summarize: " + s['document'])['input_ids']) for s in train_sample]
    summary_lengths = [len(tokenizer(s['summary'])['input_ids']) for s in train_sample]

    print("\nToken Length Analysis:")
    print(f"Document 95th percentile: {np.percentile(doc_lengths, 95):.0f}")
    print(f"Summary 95th percentile: {np.percentile(summary_lengths, 95):.0f}")
    print(f"Max_input_length={Max_input_length}, Max_target_length={Max_target_length} look reasonable.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_training()
