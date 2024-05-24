import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import load_and_clean_data
import evaluate
from evaluate import evaluator



# Load custom dataset
def load_custom_dataset(df):
    df['label'] = df['sentiment_numeric']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return DatasetDict({"train": train_dataset, "test": test_dataset})


# File path to your custom dataset

# Load the dataset
dataset = load_and_clean_data('datasets/tweets_labeled_lexicon_cleaned.json')
dataset = load_custom_dataset(dataset)
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')


# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['raw_content'], padding='max_length', truncation=True)


# Tokenize the datasets
encoded_dataset = dataset.map(tokenize_function, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load pretrained BERT model
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Load metrics
metric_precision = evaluate.load('precision')
metric_accuracy = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'precision': metric_precision.compute(predictions=predictions, references=labels)['precision']
    }

task_evaluator = evaluator("text-classification")

# evaluate.load('exact_match').compute(references=['hello'], predictions=['hello'])
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)