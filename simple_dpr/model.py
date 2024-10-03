import torch
from transformers import AutoModel, AutoTokenizer


class DPRQuestionEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DPRQuestionEncoder, self).__init__()
        # Initialize the question encoder model
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        # Forward pass through the question encoder
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the question embedding
        question_embedding = outputs.last_hidden_state[:, 0, :]
        return question_embedding


class DPRContextEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DPRContextEncoder, self).__init__()
        # Initialize the context encoder model
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        # Forward pass through the context encoder
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the context embedding
        context_embedding = outputs.last_hidden_state[:, 0, :]
        return context_embedding
