import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


class TwitterDataset(Dataset):
    def __init__(self, texts_dataset, labels_dataset, pretrained_model, sequence_length=256):
        do_lower_case = True if 'uncased' in pretrained_model else False
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lower_case)
        self.texts = texts_dataset
        self.labels = labels_dataset
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.sequence_length,
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


def create_dataloaders(df, pretrained_model, sequence_length, batch_size=16):
    texts = df['raw_content'].tolist()
    labels = df['sentiment_numeric'].tolist()

    dataset = TwitterDataset(texts, labels, pretrained_model, sequence_length)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, train_dataloader, val_dataloader, test_dataloader
