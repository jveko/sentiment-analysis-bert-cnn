import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import load_and_clean_data
import evaluate
from evaluate import evaluator



class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=768, out_channels=256, kernel_size=3):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        # self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        # self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)
        # self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        print(x)
        return x

# Extended BERT model with CNN layer
class BertWithCNN(nn.Module):
    def __init__(self, num_labels=2):
        super(BertWithCNN, self).__init__()
        self.bert = BertModel.from_pretrained('google-bert/bert-base-multilingual-uncased')
        self.cnn = CNNFeatureExtractor()
        self.fc = nn.Linear(self.cnn.out_channels, num_labels)  # Assuming binary classification


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        cnn_output = self.cnn(sequence_output)
        logits = self.fc(cnn_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

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
model = BertWithCNN(num_labels=2)

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