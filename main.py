import argparse
import wandb
import torch
import torch.nn as nn
import logging
import os
from preprocess import load_and_clean_data
from dataset import create_dataloaders
from models import BertEncoder, CNNFeatureExtractor, CombinedModel, BERT_CNN, BertOnlyModel
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

    return logger


def setup_best_model(model, uniqueId):
    if not os.path.exists('best_models'):
        os.makedirs('best_models')
    return f'best_models/best_{model}_{uniqueId}.pth'


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.model)

    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Pretrained model: {args.pretrained_model}")
    logger.info(f"Log file: logs/{args.model}-{now}.txt")
    logger.info(f"Unique ID: {uniqueId}")
    # Load and preprocess the data
    path = args.dataset
    df = load_and_clean_data(path)

    logger.info(f"Data loaded from {path}")

    # Hyperparameters CNN
    input_dim = 768
    num_filters = 256
    kernel_size = 7

    logger.info(
        f"Hyperparameters CNN: input_dim={input_dim}, num_filters={num_filters}, kernel_size={kernel_size}")

    # Hyperparameters BERT
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 20
    patience = 5
    weight_decay = 0.01
    target_names = args.neutral and target_names_3 or target_names_2

    logger.info(
        f"Hyperparameters BERT: learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}, target_names={target_names} ")
    project = "sentiment_analysis"
    if args.model == 'bert':
        project = "sentiment_analysis_bert_only"

    # Initialize wandb
    wandb.init(project=project, config={
        "model": args.model,
        "pretrained_model": args.pretrained_model,
        "dataset": path,
        "cnn": {
            "input_dim": input_dim,
            "num_filters": num_filters,
            "kernel_size": kernel_size
        },
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "weight_decay": weight_decay
    })
    # Best model path
    best_model_path = setup_best_model(args.model, uniqueId)

    # DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(df, args.pretrained_model, batch_size=batch_size)

    # Models

    if args.model == 'bert_cnn':
        model = BERT_CNN(bert_model_name=args.pretrained_model, input_dim=input_dim, num_filters=num_filters,
                         num_classes=len(target_names))
    else:
        model = BertOnlyModel(pretrained_model=args.pretrained_model)

    # if args.model == 'bert_cnn':
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model using BERT or BERT-CNN.")
    parser.add_argument('--model', choices=['bert', 'bert_cnn'], required=True,
                        help="Type of model to use: 'bert' for BERT-only model or 'bert_cnn' for BERT-CNN combined model.")
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
