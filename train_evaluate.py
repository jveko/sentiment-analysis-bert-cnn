import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score, roc_curve
import wandb
import matplotlib.pyplot as plt
import numpy as np


def train_evaluate(model, device, train_dataloader, val_dataloader, logger, config, optimizer, scheduler,
                   is_cross_validate=False, fold=None, save_model=False, save_report=False,
                   best_model_path='best_model.pth'):
    num_epochs = config["num_epochs"]
    patience = config["patience"]
    target_names = config["target_names"]
    if not is_cross_validate:
        logger.info("Training model...")
    else:
        logger.info(f"Cross Validation Fold {fold + 1}")

    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0
    best_train_loss = float('inf')
    best_epoch = 0
    best_val_epoch = 0
    no_improvement_epochs = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            if model.__class__.__name__ == 'BertForSequenceClassification':
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        if not is_cross_validate:
            wandb.log({'train_loss': avg_train_loss}, step=epoch)
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
        else:
            wandb.log({
                f"Fold {fold} Train Loss": avg_train_loss,
                f"Fold {fold} Epoch": epoch
            })
            logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                if model.__class__.__name__ == 'BertForSequenceClassification':
                    outputs = outputs.logits
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        if not is_cross_validate:
            wandb.log({'val_accuracy': accuracy}, step=epoch)
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')
        else:
            wandb.log({
                f"Fold {fold} Validation Accuracy": accuracy,
                f"Fold {fold} Epoch": epoch
            })
            logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')
        val_accuracies.append(accuracy)
        class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        if not is_cross_validate:
            if save_report:
                logger.info(f'{class_report}')

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_val_epoch = epoch + 1
                if save_model:
                    torch.save(model.state_dict(), best_model_path)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger.info('Early stopping due to lack of improvement')
                    break

        if not is_cross_validate:
            logger.info(f'Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds')
        else:
            logger.info(f'Fold {fold + 1}, Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds')

    if not is_cross_validate:
        summary = (
            f"\nTraining Summary:\n"
            f"Best Training Loss: {best_train_loss:.4f} (Epoch {best_epoch})\n"
            f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}% (Epoch {best_val_epoch})\n"
        )
    else:
        summary = (
            f"\nCross Validation Fold {fold + 1} Summary:\n"
            f"Best Training Loss: {best_train_loss:.4f} (Epoch {best_epoch})\n"
            f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}% (Epoch {best_val_epoch})\n"
        )
    logger.info(summary)

    return train_losses, val_accuracies


def train(config, model, device, dataloader, epoch, num_epochs, optimizer, scheduler, criterion, logger):
    logger.info(f"Training Epoch {epoch + 1}/{num_epochs}")
    start_time = time.time()
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        if model.__class__.__name__ == 'BertForSequenceClassification':
            outputs = outputs.logits
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Get training accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    avg_train_loss = train_loss / len(dataloader)
    wandb.log({'train_loss': avg_train_loss, "train_accuracy": accuracy}, step=epoch)
    logger.info(
        f'Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {accuracy * 100:.2f}%, Loss: {avg_train_loss:.4f}, Time: {time.time() - start_time:.2f} seconds, LR: {scheduler.get_last_lr()}')
    return avg_train_loss


def validate(config, model, device, dataloader, epoch, num_epochs, criterion, logger):
    logger.info(f"Validating Epoch {epoch + 1}/{num_epochs}")
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            if model.__class__.__name__ == 'BertForSequenceClassification':
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(dataloader)
    accuracy, conf_matrix, f1, f1_weighted, precision, precision_weighted, recall, recall_weighted = calculate_metrics(
        all_labels, all_preds)

    # For binary classification, you might want to compute ROC-AUC score
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_preds)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, ROC-AUC: {auc:.4f}')
        wandb.log({'val_roc_auc': auc}, step=epoch)

    logger.info(
        f'''Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}%, Total Loss: {avg_val_loss:.4f}
        Precision: {precision:.4f} (Weighted: {precision_weighted:.4f}), 
        Recall: {recall:.4f} (Weighted: {recall_weighted:.4f}), 
        F1 Score: {f1:.4f} (Weighted: {f1_weighted:.4f}),
        Confusion Matrix: \n{conf_matrix}''')

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=config['target_names'])
    logger.info(f'Classification Report:\n{report}')

    wandb.log({
        'val_accuracy': accuracy,
        'val_loss': avg_val_loss,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_precision_weighted': precision_weighted,
        'val_recall_weighted': recall_weighted,
        'val_f1_weighted': f1_weighted,
        "val_conf_mat": wandb.plot.confusion_matrix(probs=None, preds=all_preds, y_true=all_labels,
                                                    class_names=config['target_names'])
    }, step=epoch)
    return avg_val_loss, accuracy


def train_validate(config, model, device, train_dataloader, val_dataloader, num_epochs, optimizer, scheduler, criterion,
                   patience, save_path, logger):
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = train(config, model, device, train_dataloader, epoch, num_epochs, optimizer, scheduler, criterion,
                           logger)
        val_loss, accuracy = validate(config, model, device, val_dataloader, epoch, num_epochs, criterion, logger)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        # Break if there is no improvement in validation accuracy for 'patience' epochs
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter > patience:
                logger.info('Early stopping due to lack of improvement')
                break

    logger.info(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%')
    return train_losses, val_losses, val_accuracies


def evaluate(config, model, device, dataloader, criterion, logger):
    logger.info('Testing...')
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            if model.__class__.__name__ == 'BertForSequenceClassification':
                outputs = outputs.logits

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_total_loss = total_loss / len(dataloader)
    accuracy, conf_matrix, f1, f1_weighted, precision, precision_weighted, recall, recall_weighted = calculate_metrics(
        all_labels, all_preds)

    # Log and display additional metrics for binary classification
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_preds)
        logger.info(f'ROC-AUC: {auc:.4f}')
        wandb.log({'test_roc_auc': auc})

    logger.info(f'''
        Accuracy: {accuracy:.4f}, Total Loss: {avg_total_loss:.4f},
        Precision: {precision:.4f} (Weighted: {precision_weighted:.4f}), 
        Recall: {recall:.4f} (Weighted: {recall_weighted:.4f}), 
        F1 Score: {f1:.4f} (Weighted: {f1_weighted:.4f}),
        Confusion Matrix: \n{conf_matrix}
    ''')

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=config['target_names'])
    logger.info(f'Classification Report:\n{report}')

    wandb.log({
        'test_accuracy': accuracy,
        'test_loss': avg_total_loss,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_precision_weighted': precision_weighted,
        'test_recall_weighted': recall_weighted,
        'test_f1_weighted': f1_weighted,
        "test_conf_mat": wandb.plot.confusion_matrix(probs=None, preds=all_preds, y_true=all_labels,
                                                     class_names=config['target_names'])
    })

    return avg_total_loss, accuracy, f1, precision, recall


def calculate_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, conf_matrix, f1, f1_weighted, precision, precision_weighted, recall, recall_weighted
