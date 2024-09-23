import logging

import torch
import torch.nn.functional as F


def dpr_loss(question_embeddings, positive_embeddings, negative_embeddings):
    """
    Compute DPR loss using in-batch negatives.

    Args:
        question_embeddings (Tensor): Question embeddings (batch_size, hidden_size)
        positive_embeddings (Tensor): Positive context embeddings (batch_size, hidden_size)
        negative_embeddings (Tensor): Negative context embeddings (batch_size * num_negatives, hidden_size)

    Returns:
        Tensor: Loss value
    """
    batch_size = question_embeddings.size(0)
    question_embeddings = F.normalize(question_embeddings, p=2, dim=1)
    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
    negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

    # Concatenate positive and negative embeddings
    context_embeddings = torch.cat(
        [positive_embeddings, negative_embeddings], dim=0
    )  # (batch_size + negatives, hidden_size)

    # Compute scores and loss
    scores = torch.matmul(question_embeddings, context_embeddings.t())  # (batch_size, batch_size + negatives)
    labels = torch.arange(batch_size).to(question_embeddings.device)

    # Adjust loss calculation for multiple negatives per positive
    loss = F.cross_entropy(scores, labels)
    return loss


def train_dpr(
    question_encoder,
    context_encoder,
    train_loader,
    optimizer,
    num_epochs=3,
    device="cpu",
):
    """
    Train DPR model with provided data loader.

    Args:
        question_encoder (DPRQuestionEncoder): Question encoder model
        context_encoder (DPRContextEncoder): Context encoder model
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer for training
        num_epochs (int): Number of epochs
        device (str): Device to train on

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    question_encoder.train()
    context_encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            question_input_ids = batch["question_input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].view(-1, batch["negative_input_ids"].size(-1)).to(device)
            negative_attention_mask = (
                batch["negative_attention_mask"].view(-1, batch["negative_attention_mask"].size(-1)).to(device)
            )

            question_embeddings = question_encoder(question_input_ids, question_attention_mask)
            positive_embeddings = context_encoder(positive_input_ids, positive_attention_mask)
            negative_embeddings = context_encoder(negative_input_ids, negative_attention_mask)

            loss = dpr_loss(question_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
