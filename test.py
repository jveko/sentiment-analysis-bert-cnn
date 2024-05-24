import torch
from sklearn.metrics import accuracy_score, classification_report


def test(model, device, best_model_path, test_dataloader, logger, target_names):
    logger.info("Testing model...")

    logger.info("Loading the best model...")
    logger.info(f"Best model path: {best_model_path}")
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate the model on the test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f'Test Accuracy: {accuracy * 100:.2f}%')
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    logger.info(f'{class_report}')
