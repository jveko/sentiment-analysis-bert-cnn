import argparse
import wandb
import torch
import logging
import os
from preprocess import load_and_clean_data
from dataset import create_dataloaders
from models import BERT_CNN, BertOnlyModel, BERT_LSTM, BERT_GRU
from datetime import datetime
from train import train
from nanoid import generate
from pytz import timezone

tz = timezone('Asia/Jakarta')
os.environ['TZ'] = 'Asia/Jakarta'
now = datetime.now(tz=tz).strftime('%Y-%m-%d_%H:%M:%S')
uniqueId = generate('abcdefghijklmnopqrstuvwxyz', 10)
wandb.login()
target_names_2 = ['negative', 'positive']
target_names_3 = ['negative', 'positive', 'neutral']


def timetz(*args):
    return datetime.now(tz).timetuple()


def setup_logging(model):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file = f'logs/{model}-{now}.txt'
    logging.Formatter.converter = timetz
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, file_handler


def setup_best_model(model, uniqueId):
    if not os.path.exists('best_models'):
        os.makedirs('best_models')
    return f'best_models/best_{model}_{uniqueId}.pth'


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    learning_rate = 2e-5
    batch_size = 32
    num_epochs = 10
    patience = 3
    weight_decay = 0.01
    target_names = args.neutral and target_names_3 or target_names_2

    path = args.dataset
    df = load_and_clean_data(path)
    train_dataloader, val_dataloader = create_dataloaders(df, args.pretrained_model, batch_size=batch_size)
    print(args.models)
    for args_model in args.models:
        logger, file_handler = setup_logging(args_model)

        logger.info(f"Device: {device}")
        logger.info(f"Model: {args_model}")
        logger.info(f"Pretrained model: {args.pretrained_model}")
        logger.info(f"Log file: logs/{args_model}-{now}.txt")
        logger.info(f"Unique ID: {uniqueId}")

        logger.info(f"Data loaded from {path}")

        logger.info(
            f"Hyperparameters BERT: learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}, target_names={target_names} ")

        project = f"sentiment_analysis_{args_model}"
        config = {
            "model": args_model,
            "pretrained_model": args.pretrained_model,
            "dataset": path,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "weight_decay": weight_decay
        }

        # Best model path
        best_model_path = setup_best_model(args_model, uniqueId)

        # Models

        if args_model == 'bert_cnn':
            # Hyperparameters CNN
            input_dim = 768
            num_filters = 256
            kernel_size = 7

            config["CNN"] = {
                "input_dim": input_dim,
                "num_filters": num_filters,
                "kernel_size": kernel_size
            }

            logger.info(
                f"Hyperparameters CNN: input_dim={input_dim}, num_filters={num_filters}, kernel_size={kernel_size}")
            model = BERT_CNN(bert_model_name=args.pretrained_model, input_dim=input_dim, num_filters=num_filters,
                             num_classes=len(target_names))
        elif args_model == 'bert_lstm':
            # Hyperparameters CNN
            input_dim = 768
            hidden_dim = 256
            bidirectional = False
            dropout_rate = 0.2

            config["LSTM"] = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "dropout_rate": dropout_rate,
            }

            logger.info(
                f"Hyperparameters LSTM: input_dim={input_dim}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate}")
            model = BERT_LSTM(bert_model_name=args.pretrained_model, input_dim=input_dim, hidden_dim=hidden_dim,
                              num_layers=3, num_classes=len(target_names), dropout_rate=dropout_rate,
                              bidirectional=bidirectional)
        elif args_model == 'bert_gru':
            # Hyperparameters CNN
            input_dim = 768
            hidden_dim = 256
            bidirectional = False
            dropout_rate = 0.2

            config["GRU"] = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "dropout_rate": dropout_rate,
            }
            logger.info(
                f"Hyperparameters GRU: input_dim={input_dim}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate}")
            model = BERT_GRU(bert_model_name=args.pretrained_model, input_dim=input_dim, hidden_dim=hidden_dim,
                             num_layers=3, num_classes=len(target_names), dropout_rate=dropout_rate,
                             bidirectional=bidirectional)
        else:
            model = BertOnlyModel(pretrained_model=args.pretrained_model)

        # Initialize wandb
        wandb.init(project=project, config=config)
        wandb.run.log_code(".")

        # if args_model == 'bert_cnn':
        #     def initialize_weights(m):
        #         if isinstance(m, (nn.Conv1d, nn.Linear)):
        #             nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)

        # model.apply(initialize_weights)

        model.to(device)

        # Training
        train(model=model, device=device, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
              num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, weight_decay=weight_decay,
              best_model_path=best_model_path, logger=logger, target_names=target_names, save_model=args.save_model,
              save_report=args.report)

        # Testing
        # if args.save_model:
        #     test(model=model, device=device, best_model_path=best_model_path, test_dataloader=val_dataloader,
        #          logger=logger, target_names=target_names)

        wandb.finish()

        logger.removeHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis with BERT and CNN, LSTM, GRU.")
    parser.add_argument('--models', choices=['bert', 'bert_cnn', 'bert_lstm', 'bert_gru'], required=True, nargs='+',
                        help="Type of models")
    parser.add_argument('--pretrained_model', default='bert-base-multilingual-cased',
                        help="Pretrained model name or path.")
    parser.add_argument('--dataset', default='datasets/tweets_labeled_lexicon.json',
                        help="Path to the dataset file. Default: 'datasets/tweets_labeled_lexicon.json'.")
    parser.add_argument('--neutral', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to include neutral class in the classification report.")
    parser.add_argument('--save-model', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to save the best model.")
    parser.add_argument('--report', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to include classification report in the log file.")
    args = parser.parse_args()

    main(args)
