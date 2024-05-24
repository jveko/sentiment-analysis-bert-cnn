import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


class TwitterDataset1(Dataset):
    def __init__(self, texts_dataset, labels_dataset, tokenizer):
        self.texts = texts_dataset
        self.labels = labels_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TwitterDataset(Dataset):
    def __init__(self, texts_dataset, labels_dataset, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.texts = texts_dataset
        self.labels = labels_dataset

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(df, pretrained_model, batch_size=16, val_split=0.2):
    texts = df['raw_content'].tolist()
    labels = df['sentiment_numeric'].tolist()

    dataset = TwitterDataset(texts, labels, pretrained_model)

    train_size = int((1-val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
