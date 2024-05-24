import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import wandb


def train(model, device, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, weight_decay,
          best_model_path, logger,target_names, save_model=False, save_report=False):
    logger.info("Training model...")

    wandb.watch(model, log="all")

    # Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # TensorBoard
    writer = SummaryWriter(log_dir='runs/BERT_CNN_SA')

    best_val_accuracy = 0
    best_train_loss = float('inf')
    best_epoch = 0
    best_val_epoch = 0
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        wandb.log({'train_loss': avg_train_loss}, step=epoch)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

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
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        wandb.log({'val_accuracy': accuracy}, step=epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')

        class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
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

        logger.info(f'Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds')

    writer.close()

    summary = (
        f"\nTraining Summary:\n"
        f"Best Training Loss: {best_train_loss:.4f} (Epoch {best_epoch})\n"
        f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}% (Epoch {best_val_epoch})\n"
    )
    logger.info(summary)
