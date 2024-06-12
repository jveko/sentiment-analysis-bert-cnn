import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERT_CNN(nn.Module):
    def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, num_filters=256,
                 kernel_sizes=None, sequence_length=128, num_classes=2, dropout_rate=0.2):
        super(BERT_CNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_filters * len(kernel_sizes))
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 128)  # Add one more fully connected layer
        self.fc2 = nn.Linear(128, num_classes)
        self._init_weights()

        # self.flattened_dim = num_filters * len(kernel_sizes) * (sequence_length // 2)
        # self.fc1 = nn.Linear(self.flattened_dim, num_filters)
        # # self.fc1 = nn.Linear(num_filters * len(kernel_sizes), num_classes)  # 3 * 256 because of three conv layers
        # self.fc2 = nn.Linear(num_filters, num_classes)

    def _init_weights(self):
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # Transpose to (batch_size, 768, seq_len) for Conv1d
        # x = x.transpose(1, 2)
        x = x.permute(0, 2, 1)
        # conv_outs = [torch.max(self.pool(F.relu(conv(x))), 2)[0] for conv in self.conv_layers]
        # conv_outs = [self.pool(F.relu(conv(x))) for conv in self.conv_layers]
        conv_outs = [self.global_max_pool(F.relu(conv(x))).squeeze(-1) for conv in self.conv_layers]
        x = torch.cat(conv_outs, 1)
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)

        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # Added ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# class BERT_LSTM(nn.Module):
#     def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, hidden_dim=256,
#                  num_layers=2, num_classes=2, dropout_rate=0.2, bidirectional=True):
#         super(BERT_LSTM, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True,
#                             dropout=dropout_rate)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
#
#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
#         # x = self.dropout(x)
#         lstm_out, (hidden, cell) = self.lstm(x)
#
#         if self.lstm.bidirectional:
#             hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         else:
#             hidden = hidden[-1, :, :]
#
#         x = self.dropout(hidden)
#         x = self.fc(x)
#
#         return x
class BERT_LSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-multilingual-uncased',
                 input_dim=768, hidden_dim=256, num_layers=2,
                 num_classes=2, dropout_rate=0.5, bidirectional=True):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)

        lstm_out, (hidden, cell) = self.lstm(x)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.layer_norm(hidden)
        hidden = F.relu(hidden)
        x = self.dropout(hidden)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


# class BERT_GRU(nn.Module):
#     def __init__(self, bert_model_name='google-bert/bert-base-multilingual-uncased', input_dim=768, hidden_dim=256,
#                  num_layers=2, num_classes=2, dropout_rate=0.2, bidirectional=True):
#         super(BERT_GRU, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True,
#                           dropout=dropout_rate)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
#
#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
#         # x = self.dropout(x)
#         gru_out, hidden = self.gru(x)
#
#         if self.gru.bidirectional:
#             hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         else:
#             hidden = hidden[-1, :, :]
#
#         x = self.dropout(hidden)
#         x = self.fc(x)
#
#         return x

class BERT_GRU(nn.Module):
    def __init__(self, bert_model_name='bert-base-multilingual-uncased',
                 input_dim=768, hidden_dim=256, num_layers=2,
                 num_classes=2, dropout_rate=0.5, bidirectional=True):
        super(BERT_GRU, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)

        gru_out, hidden = self.gru(x)

        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.layer_norm(hidden)
        hidden = F.relu(hidden)
        x = self.dropout(hidden)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
