import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from simple_dpr.evaluate import evaluate
from simple_dpr.model import DPRContextEncoder, DPRQuestionEncoder
from simple_dpr.train import train_dpr


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_dpr.log")],
    )


def drop_insufficient_data(df):
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[id_df.exact > 0]
    id_df = id_df[id_df.exact != id_df.total]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def load_data():
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


class QueryDocumentTripletDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, num_negatives=4, size=0):
        """
        Dataset for query, positive, and negative triplet samples.
        Args:
            df (DataFrame): DataFrame containing query and document information.
            tokenizer (AutoTokenizer): Tokenizer to encode the queries and documents.
            max_length (int): Maximum length for tokenization.
            num_negatives (int): Number of negative contexts per example
            size (int): Subset size (if > 0, limits the dataset size).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.queries = df.groupby("query_id")
        self.query_ids = list(self.queries.groups.keys())
        if size > 0:
            self.query_ids = self.query_ids[:size]

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_group = self.queries.get_group(self.query_ids[idx])
        query = query_group.iloc[0]["query"]

        positive_sample = query_group[query_group["exact"] == 1]
        if positive_sample.empty:
            raise ValueError(f"No positive samples for query_id: {self.query_ids[idx]}")
        positive_doc = positive_sample.sample(1).iloc[0]["product_title"]

        negative_sample = query_group[query_group["exact"] == 0]
        if negative_sample.empty:
            raise ValueError(f"No negative samples for query_id: {self.query_ids[idx]}")
        # Always get `num_negatives` negative samples. If there are not enough, sample with replacement.
        negative_docs = negative_sample.sample(self.num_negatives, replace=len(negative_sample) < self.num_negatives)[
            "product_title"
        ].tolist()

        question_encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        positive_encoding = self.tokenizer(
            positive_doc,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        negative_encodings = [
            self.tokenizer(
                negative_doc,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            for negative_doc in negative_docs
        ]

        return {
            "question_input_ids": question_encoding["input_ids"].squeeze(0),
            "question_attention_mask": question_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": torch.stack(
                [negative_encoding["input_ids"].squeeze(0) for negative_encoding in negative_encodings]
            ),
            "negative_attention_mask": torch.stack(
                [negative_encoding["attention_mask"].squeeze(0) for negative_encoding in negative_encodings]
            ),
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle the variable negative sample size in each batch.

    Args:
        batch (list): List of samples fetched by Dataset.

    Returns:
        dict: A dictionary with properly collated tensors.
    """
    question_input_ids = torch.stack([item["question_input_ids"] for item in batch])
    question_attention_mask = torch.stack([item["question_attention_mask"] for item in batch])
    positive_input_ids = torch.stack([item["positive_input_ids"] for item in batch])
    positive_attention_mask = torch.stack([item["positive_attention_mask"] for item in batch])
    negative_input_ids = torch.cat([item["negative_input_ids"] for item in batch], dim=0)
    negative_attention_mask = torch.cat([item["negative_attention_mask"] for item in batch], dim=0)

    return {
        "question_input_ids": question_input_ids,
        "question_attention_mask": question_attention_mask,
        "positive_input_ids": positive_input_ids,
        "positive_attention_mask": positive_attention_mask,
        "negative_input_ids": negative_input_ids,
        "negative_attention_mask": negative_attention_mask,
    }


def save_models(question_encoder, context_encoder, optimizer, save_directory="dpr_model"):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    question_encoder.model.save_pretrained(os.path.join(save_directory, "question_encoder"))
    context_encoder.model.save_pretrained(os.path.join(save_directory, "context_encoder"))
    question_encoder.tokenizer.save_pretrained(save_directory)

    optimizer_path = os.path.join(save_directory, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Loading data from Amazon ESCI dataset...")
    train_df, test_df = load_data()
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    logger.info("Initializing tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    question_encoder = DPRQuestionEncoder(model_name="bert-base-uncased")
    context_encoder = DPRContextEncoder(model_name="bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_encoder.to(device)
    context_encoder.to(device)

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentTripletDataset(train_df, tokenizer, num_negatives=4)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    test_dataset = QueryDocumentTripletDataset(test_df, tokenizer, num_negatives=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

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
    save_models(question_encoder, context_encoder, optimizer)

    logger.info("Evaluating the model...")
    evaluate(
        question_encoder,
        context_encoder,
        test_loader,
        device=device,
    )
