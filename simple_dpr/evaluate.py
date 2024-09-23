import logging

import torch
import torch.nn.functional as F


def evaluate(
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

    with torch.no_grad():
        for batch in dataloader:
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
            total_loss += loss.item()

            # Updated accuracy calculation logic
            batch_size = question_embeddings.size(0)
            context_embeddings = torch.cat(
                [positive_embeddings] + torch.split(negative_embeddings, batch_size, dim=0), dim=0
            )
            scores = torch.matmul(question_embeddings, context_embeddings.t())
            # Creating the correct labels for the current batch size
            preds = torch.argmax(scores, dim=1)
            labels = torch.cat(
                [torch.zeros(batch_size, dtype=torch.long)]
                + [torch.full((batch_size,), i + 1, dtype=torch.long) for i in range(len(scores) // batch_size - 1)],
                dim=0,
            ).to(device)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
