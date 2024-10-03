import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from simple_dpr.evaluate import evaluate_dpr
from simple_dpr.model import DPRContextEncoder, DPRQuestionEncoder
from simple_dpr.train import train_dpr


def setup_logger():
    """
    Set up the logger for logging training and evaluation information.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_dpr.log")],
    )


def drop_insufficient_data(df):
    """
    Drop queries that do not have enough positive examples.

    Args:
        df (DataFrame): DataFrame containing query and document information.

    Returns:
        DataFrame: Filtered DataFrame with sufficient positive examples.
    """
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[id_df.exact > 0]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def load_data():
    """
    Load and preprocess the Amazon ESCI dataset.

    Returns:
        DataFrame: Training DataFrame.
        DataFrame: Test DataFrame.
    """
    product_df = pd.read_parquet("downloads/shopping_queries_dataset_products.parquet")
    example_df = pd.read_parquet("downloads/shopping_queries_dataset_examples.parquet")
    df = pd.merge(
        example_df[["example_id", "query_id", "product_id", "query", "esci_label", "split"]],
        product_df[["product_id", "product_title"]],
        how="left",
        on="product_id",
    )[["example_id", "query_id", "query", "product_title", "esci_label", "split"]]
    df["exact"] = df.esci_label.apply(lambda x: 1 if x == "E" else 0)
    train_df = drop_insufficient_data(
        df[df.split == "train"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    test_df = drop_insufficient_data(
        df[df.split == "test"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    return train_df, test_df


class QueryDocumentDataset(Dataset):
    def __init__(self, df, tokenizer, train=True, max_length=128):
        """
        Dataset for query and positive document pairs.

        Args:
            df (DataFrame): DataFrame containing query and positive document information.
            tokenizer (AutoTokenizer): Tokenizer to encode the queries and documents.
            train (bool): If True, use only positive examples.
            max_length (int): Maximum length for tokenization.
        """
        if train:
            self.df = df[df["exact"] == 1]
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query = self.df.loc[idx, "query"]
        context_doc = self.df.loc[idx, "product_title"]
        target = self.df.loc[idx, "exact"]

        # Encode the query
        question_encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Encode the positive document
        context_encoding = self.tokenizer(
            context_doc,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "question_input_ids": question_encoding["input_ids"].squeeze(0),
            "question_attention_mask": question_encoding["attention_mask"].squeeze(0),
            "context_input_ids": context_encoding["input_ids"].squeeze(0),
            "context_attention_mask": context_encoding["attention_mask"].squeeze(0),
            "target": target,
        }


def save_models(question_encoder, context_encoder, optimizer, tokenizer, save_directory="dpr_model"):
    """
    Save the trained models and optimizer state.

    Args:
        question_encoder (DPRQuestionEncoder): Trained question encoder model.
        context_encoder (DPRContextEncoder): Trained context encoder model.
        optimizer (Optimizer): Optimizer used during training.
        tokenizer (AutoTokenizer): Tokenizer used for encoding.
        save_directory (str): Directory to save the models and optimizer.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the encoder models
    question_encoder.model.save_pretrained(os.path.join(save_directory, "question_encoder"))
    context_encoder.model.save_pretrained(os.path.join(save_directory, "context_encoder"))

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)

    # Save the optimizer state
    optimizer_path = os.path.join(save_directory, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)


def load_models(logger, model_name="bert-base-uncased", save_directory="dpr_model", device="cpu"):
    """
    Load the trained models and optimizer state.

    Args:
        logger (Logger): Logger for logging information.
        model_name (str): Name of the pretrained model.
        save_directory (str): Directory where models are saved.
        device (str): Device to load the models onto.

    Returns:
        DPRQuestionEncoder: Loaded question encoder model.
        DPRContextEncoder: Loaded context encoder model.
        Optimizer: Loaded optimizer state.
        AutoTokenizer: Loaded tokenizer.
    """
    logger.info("Loading models...")
    question_encoder = DPRQuestionEncoder(model_name=model_name)
    context_encoder = DPRContextEncoder(model_name=model_name)

    # Load the pretrained encoder models
    question_encoder.model = question_encoder.model.from_pretrained(os.path.join(save_directory, "question_encoder"))
    context_encoder.model = context_encoder.model.from_pretrained(os.path.join(save_directory, "context_encoder"))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    question_encoder.to(device)
    context_encoder.to(device)

    # Initialize optimizer and load its state
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=2e-5)
    optimizer.load_state_dict(torch.load(os.path.join(save_directory, "optimizer.pt"), map_location=device))

    return question_encoder, context_encoder, optimizer, tokenizer


def train(logger, train_df, model_name="bert-base-uncased", device="cpu"):
    """
    Train the DPR model.

    Args:
        logger (Logger): Logger for logging information.
        train_df (DataFrame): Training DataFrame.
        model_name (str): Name of the pretrained model.
        device (str): Device to train on.

    Returns:
        None
    """
    logger.info("Initializing tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question_encoder = DPRQuestionEncoder(model_name=model_name)
    context_encoder = DPRContextEncoder(model_name=model_name)

    question_encoder.to(device)
    context_encoder.to(device)

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentDataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=2e-5)

    logger.info("Starting training...")
    train_dpr(
        question_encoder,
        context_encoder,
        train_loader,
        optimizer,
        num_epochs=3,
        device=device,
    )

    logger.info("Saving trained models...")
    save_models(question_encoder, context_encoder, optimizer, tokenizer)


def evaluate(logger, test_df, model_name="bert-base-uncased", device="cpu"):
    """
    Evaluate the trained DPR model.

    Args:
        logger (Logger): Logger for logging information.
        test_df (DataFrame): Test DataFrame.
        model_name (str): Name of the pretrained model.
        device (str): Device to evaluate on.

    Returns:
        None
    """
    logger.info("Evaluating the model...")

    # Load models and tokenizer
    question_encoder, context_encoder, _, tokenizer = load_models(logger, model_name=model_name, device=device)

    test_dataset = QueryDocumentDataset(test_df, tokenizer, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info("Evaluating the model...")
    avg_loss, accuracy = evaluate_dpr(
        question_encoder,
        context_encoder,
        test_loader,
        device=device,
    )
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"

    logger.info("Loading data from Amazon ESCI dataset...")
    train_df, test_df = load_data()
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    train(logger, train_df, model_name=model_name, device=device)

    evaluate(logger, test_df, model_name=model_name, device=device)
