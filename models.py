import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERT_CNN(nn.Module):
    def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, num_filters=256,
                 kernel_sizes=[3, 4, 5], num_classes=2, dropout_rate=0.2):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)  # 3 * 256 because of three conv layers
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # x = self.dropout(x)
        # Transpose to (batch_size, 768, seq_len) for Conv1d
        x = x.transpose(1, 2)
        conv_outs = [torch.max(self.pool(F.relu(conv(x))), 2)[0] for conv in self.conv_layers]
        x = torch.cat(conv_outs, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class BERT_LSTM(nn.Module):
    def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout_rate=0.2, bidirectional=True):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True,
                            dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # x = self.dropout(x)
        lstm_out, (hidden, cell) = self.lstm(x)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        x = self.dropout(hidden)
        x = self.fc(x)

        return x


class BERT_GRU(nn.Module):
    def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout_rate=0.2, bidirectional=True):
        super(BERT_GRU, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True,
                          dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # x = self.dropout(x)
        gru_out, hidden = self.gru(x)

        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        x = self.dropout(hidden)
        x = self.fc(x)

        return x


class BertOnlyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(BertOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x.squeeze(-1))
        return x
