from models import BERT_CNN, BERT_LSTM, BERT_GRU
from transformers import BertForSequenceClassification


def get_model(config, logger):
    if config['model'] == 'bert_cnn':
        # Hyperparameters CNN
        input_dim = 768
        num_filters = 128
        kernel_sizes = [3, 4, 5]
        dropout_rate = 0.2

        config["CNN"] = {
            "input_dim": input_dim,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "dropout_rate": dropout_rate,
        }

        logger.info(
            f"Hyperparameters CNN: input_dim={input_dim}, num_filters={num_filters}, kernel_sizes={kernel_sizes} dropout_rate={dropout_rate}")
        model = BERT_CNN(bert_model_name=config['pretrained_model'], input_dim=input_dim, num_filters=num_filters,
                         kernel_sizes=kernel_sizes, num_classes=len(config['target_names']),
                         sequence_length=config['sequence_length'], dropout_rate=dropout_rate)
    elif config['model'] == 'bert_lstm':
        # Hyperparameters CNN
        input_dim = 768
        hidden_dim = 128
        bidirectional = False
        dropout_rate = 0.5
        num_layers = 2

        config["LSTM"] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
        }

        logger.info(
            f"Hyperparameters LSTM: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout_rate={dropout_rate}")
        model = BERT_LSTM(bert_model_name=config['pretrained_model'], input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, num_classes=len(config['target_names']), dropout_rate=dropout_rate,
                          bidirectional=bidirectional)
    elif config['model'] == 'bert_gru':
        # Hyperparameters CNN
        input_dim = 768
        hidden_dim = 128
        bidirectional = False
        dropout_rate = 0.5
        num_layers = 2

        config["GRU"] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
        }
        logger.info(
            f"Hyperparameters GRU: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout_rate={dropout_rate}")
        model = BERT_GRU(bert_model_name=config['pretrained_model'], input_dim=input_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers, num_classes=len(config['target_names']), dropout_rate=dropout_rate,
                         bidirectional=bidirectional)
    elif config['model'] == 'bert_bilstm':
        # Hyperparameters CNN
        input_dim = 768
        hidden_dim = 128
        bidirectional = True
        dropout_rate = 0.5
        num_layers = 2

        config["BiLSTM"] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
        }
        logger.info(
            f"Hyperparameters BiLSTM: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout_rate={dropout_rate}")
        model = BERT_LSTM(bert_model_name=config['pretrained_model'], input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, num_classes=len(config['target_names']), dropout_rate=dropout_rate,
                          bidirectional=bidirectional)
    elif config['model'] == 'bert_bigru':
        # Hyperparameters CNN
        input_dim = 768
        hidden_dim = 128
        bidirectional = True
        dropout_rate = 0.5
        num_layers = 2

        config["BiGRU"] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
        }
        logger.info(
            f"Hyperparameters BiGRU: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout_rate={dropout_rate}")
        model = BERT_GRU(bert_model_name=config['pretrained_model'], input_dim=input_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers, num_classes=len(config['target_names']), dropout_rate=dropout_rate,
                         bidirectional=bidirectional)
    else:
        model = BertForSequenceClassification.from_pretrained(config['pretrained_model'],
                                                              num_labels=len(config['target_names']))
    return model
