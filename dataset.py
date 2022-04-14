import torch
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

"""
"id": "593f14f960d971e294af884f0194b3a7",
"question": "舍本和誰的數據能推算出連星的恆星的質量？",
"paragraphs": [
    2018,
    6952,
    8264,
    836
],
"relevant": 836,
"answer": {
    "text": "斯特魯維",
    "start": 108
}
"""

class contextDataset(Dataset):
    def __init__(self, context, data, tokenizer, max_len, split):
        self.context = context
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess(split)

    def preprocess(self, split):
        self.tokenized_examples = []
        for examples in tqdm(self.data):
            # preprocess
            question_headers = [examples["question"] for j in range(4)]
            # question_headers = [[context] * 4 for context in examples["sent1"]]
            # sentences = [
            #     [f"{question_headers} {self.context[p]}" for p in examples["paragraphs"]]
            # ]
            sentences = \
                [self.context[p] for p in examples["paragraphs"]]

            tokenized_example = self.tokenizer(question_headers, sentences, padding=True, truncation=True, max_length=self.max_len)
            self.tokenized_examples.append({k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_example.items()})
            self.tokenized_examples[-1]["labels"] = examples["paragraphs"].index(examples["relevant"])
            
        # return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    def __len__(self) -> int:
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]
