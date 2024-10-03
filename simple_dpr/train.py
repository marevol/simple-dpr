import logging

import torch
import torch.nn.functional as F


def dpr_loss(question_embeddings, context_embeddings):
    """
    Compute DPR loss using in-batch negatives.

    Args:
        question_embeddings (Tensor): Question embeddings (batch_size, hidden_size)
        context_embeddings (Tensor): Positive context embeddings (batch_size, hidden_size)

    Returns:
        Tensor: Loss value
    """
    # Normalize embeddings
    question_embeddings = F.normalize(question_embeddings, p=2, dim=1)
    context_embeddings = F.normalize(context_embeddings, p=2, dim=1)

    # Compute similarity scores between all question and positive pairs (batch_size x batch_size)
    scores = torch.matmul(question_embeddings, context_embeddings.t())

    # Targets are diagonal elements (correct pairs)
    targets = torch.arange(scores.size(0)).long().to(question_embeddings.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(scores, targets)
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
    Train DPR model with in-batch negatives.

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

    for epoch in range(num_epochs):
        question_encoder.train()
        context_encoder.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            question_input_ids = batch["question_input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)

            # Encode questions and positive contexts
            question_embeddings = question_encoder(question_input_ids, question_attention_mask)
            context_embeddings = context_encoder(context_input_ids, context_attention_mask)

            # Compute loss using in-batch negatives
            loss = dpr_loss(question_embeddings, context_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
