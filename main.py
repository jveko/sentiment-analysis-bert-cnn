import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import os

from get_model import get_model
from preprocess import load_and_clean_data
from dataset import create_dataloaders
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
from train_evaluate import train_validate, evaluate
from nanoid import generate
from pytz import timezone

tz = timezone('Asia/Jakarta')
os.environ['TZ'] = 'Asia/Jakarta'
now = datetime.now(tz=tz).strftime('%Y-%m-%d_%H:%M:%S')
uniqueId = generate('abcdefghijklmnopqrstuvwxyz', 10)
wandb.login()
target_names_2 = ['negative', 'positive']
target_names_3 = ['negative', 'neutral', 'positive']


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    learning_rate = 2e-5
    batch_size = 32
    num_epochs = 20
    patience = 3
    sequence_length = 128
    target_names = args.neutral and target_names_3 or target_names_2

    path = args.dataset
    df = load_and_clean_data(path, target_names=target_names)
    dataset, train_dataloader, val_dataloader, test_dataloader = create_dataloaders(df, args.pretrained_model,
                                                                                    batch_size=batch_size,
                                                                                    sequence_length=sequence_length)
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
            "dataset": {
                "path": path,
                "num_samples": len(df),
                "num_classes": len(target_names),
            },
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "target_names": target_names,
        }

        # Models
        model = get_model(config, logger)

        # Initialize wandb
        wandb.init(project=project, config=config)
        wandb.run.log_code(".")
        wandb.watch(model, log="all")

        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("train_accuracy", summary="max")

        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_accuracy", summary="max")
        wandb.define_metric("val_f1", summary="max")
        wandb.define_metric("val_precision", summary="max")
        wandb.define_metric("val_recall", summary="max")
        wandb.define_metric("val_roc_auc", summary="max")
        wandb.define_metric("val_precision_weighted", summary="max")
        wandb.define_metric("val_recall_weighted", summary="max")
        wandb.define_metric("val_f1_weighted", summary="max")

        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_accuracy", summary="max")
        wandb.define_metric("test_f1", summary="max")
        wandb.define_metric("test_precision", summary="max")
        wandb.define_metric("test_recall", summary="max")
        wandb.define_metric("test_roc_auc", summary="max")
        wandb.define_metric("test_precision_weighted", summary="max")
        wandb.define_metric("test_recall_weighted", summary="max")
        wandb.define_metric("test_f1_weighted", summary="max")

        model.to(device)

        # Optimizer and scheduler
        if args_model == 'bert':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            weight_decay = 0.01
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate / 3.0},
                {'params': [p for n, p in model.bert.encoder.layer[:8].named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate / 2.0},
                {'params': [p for n, p in model.bert.encoder.layer[8:].named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate},
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n],
                 'weight_decay': weight_decay, 'lr': learning_rate * 10},
            ]
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()
        save_path = f"{args_model}.pt"

        # if args_model == 'bert':
        #     evaluate(model=model, device=device, dataloader=test_dataloader, criterion=criterion, logger=logger)
        #     wandb.finish()
        #     logger.removeHandler(file_handler)
        #     continue

        train_validate(config=config, model=model, device=device, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, num_epochs=config["num_epochs"], optimizer=optimizer,
                       scheduler=scheduler, criterion=criterion, patience=patience, save_path=save_path, logger=logger)

        model.load_state_dict(torch.load(save_path))

        # Testing
        evaluate(config=config, model=model, device=device, dataloader=test_dataloader, criterion=criterion,
                 logger=logger)

        wandb.log({"dataset": wandb.Table(dataframe=df)})
        wandb.log({"dataset_distribution_table": wandb.Table(dataframe=df['sentiment'].value_counts().reset_index())})
        wandb.log({"train_dataloader": wandb.Table(dataframe=df.iloc[train_dataloader.dataset.indices])})
        wandb.log({"val_dataloader": wandb.Table(dataframe=df.iloc[val_dataloader.dataset.indices])})
        wandb.log({"test_dataloader": wandb.Table(dataframe=df.iloc[test_dataloader.dataset.indices])})
        wandb.log({"model_summary": str(model)})
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb.finish()
        logger.removeHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis with BERT and CNN, LSTM, GRU.")
    parser.add_argument('--models', choices=['bert', 'bert_cnn', 'bert_lstm', 'bert_gru', 'bert_bilstm', 'bert_bigru'],
                        required=True, nargs='+',
                        help="Type of models")
    parser.add_argument('--pretrained_model', default='bert-base-multilingual-cased',
                        help="Pretrained model name or path.")
    parser.add_argument('--dataset', default='datasets/tweets_labeled_lexicon.json',
                        help="Path to the dataset file. Default: 'datasets/tweets_labeled_lexicon.json'.")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to train the model.")
    parser.add_argument('--neutral', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to include neutral class in the classification report.")
    args = parser.parse_args()

    main(args)
