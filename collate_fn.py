import copy
import torch
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    split: str = "train"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        features_cpy = copy.deepcopy(features)
        if self.split == "train":
            label_name = "labels"
            labels = [feature.pop(label_name) for feature in features_cpy]
        batch_size = len(features_cpy)
        num_choices = len(features_cpy[0]["input_ids"][0])
        flattened_features = [
            [{k: v[0][i] for k, v in feature.items()} for i in range(num_choices)] for feature in features_cpy
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if self.split == "train":
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

@dataclass
class DataCollatorForQuestionAnswering:
    """
    Data collator for quesion answering.
    """

    def __call__(self, features):
        first = features[0]
        batch = {}

        for k, v in first.items():
            if k not in ("label", "label_ids") and not (None in v) and not isinstance(v[0], str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
        return batch