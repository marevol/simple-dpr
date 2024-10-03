import logging

import torch
import torch.nn.functional as F


def evaluate_dpr(
    question_encoder,
    context_encoder,
    dataloader,
    device="cpu",
):
    """
    Evaluate DPR model on validation dataset.

    Args:
        question_encoder (DPRQuestionEncoder): Question encoder model
        context_encoder (DPRContextEncoder): Context encoder model
        dataloader (DataLoader): Validation data loader
        device (str): Device to evaluate on

    Returns:
        float: Average loss
        float: Accuracy
    """
    logger = logging.getLogger(__name__)
    question_encoder.eval()
    context_encoder.eval()
    total_loss = 0
    correct = 0
    total = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            question_input_ids = batch["question_input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            targets = batch["target"].float().to(device)  # Convert targets to float for BCEWithLogitsLoss

            # Encode questions and contexts
            question_embeddings = question_encoder(question_input_ids, question_attention_mask)
            context_embeddings = context_encoder(context_input_ids, context_attention_mask)

            # Normalize embeddings
            question_embeddings = F.normalize(question_embeddings, p=2, dim=1)
            context_embeddings = F.normalize(context_embeddings, p=2, dim=1)

            # Compute similarity scores (dot product per sample)
            scores = (question_embeddings * context_embeddings).sum(dim=1)  # (batch_size,)

            # Compute loss
            loss = criterion(scores, targets)
            total_loss += loss.item()

            # Compute predictions
            preds = (torch.sigmoid(scores) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
