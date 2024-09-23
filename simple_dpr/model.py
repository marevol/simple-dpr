import torch
from transformers import AutoModel, AutoTokenizer


class DPRQuestionEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DPRQuestionEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Obtain the [CLS] token representation for the question
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output


class DPRContextEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DPRContextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Obtain the [CLS] token representation for the context
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output
