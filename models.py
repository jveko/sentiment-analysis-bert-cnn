import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
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
        return x


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
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # Transpose to (batch_size, 768, seq_len) for Conv1d
        x = sequence_output.transpose(1, 2)
        conv_outs = [torch.max(self.pool(F.relu(conv(x))), 2)[0] for conv in self.conv_layers]
        x = torch.cat(conv_outs, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BertOnlyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(BertOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        x = self.bert_encoder(input_ids, attention_mask)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x.squeeze(-1))
        return x


class CombinedModel(nn.Module):
    def __init__(self, bert_encoder_model, cnn_feature_extractor_model, num_classes):
        super(CombinedModel, self).__init__()
        self.bert_encoder = bert_encoder_model
        self.cnn_feature_extractor = cnn_feature_extractor_model
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(cnn_feature_extractor_model.out_channels, num_classes)  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        x = self.bert_encoder(input_ids, attention_mask)
        x = self.cnn_feature_extractor(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x
