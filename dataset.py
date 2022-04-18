from tqdm import tqdm
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
class QADataset(Dataset):
    def __init__(self, context, data, tokenizer, max_len, split, relevant=None, preprocess=True):
        self.context = context
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.preprocess = preprocess
        if self.preprocess:
            if self.split == 'train':
                self.prepare_train_features()
            else:
                self.prepare_validation_features(relevant)
        else:
            if relevant is None:
                for idx, examples in enumerate(self.data):
                    examples["context"] = self.context[examples["relevant"]]
            else:
                for idx, examples in enumerate(self.data):
                    examples["context"] = self.context[examples["paragraphs"][relevant[idx]]]

    # Training preprocessing
    def prepare_train_features(self):
        self.inputs = []
        for examples in self.data:
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples["question"] = examples["question"].lstrip()

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = self.tokenizer(
                examples["question"],
                self.context[examples["relevant"]],
                truncation="only_second",
                max_length=self.max_len,
                # stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            answers = examples['answer']
            for i, offsets in enumerate(offset_mapping[:1]):
                tokenized_example = {}
                # Let's label those examples!
                tokenized_example["start_positions"] = []
                tokenized_example["end_positions"] = []

                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                tokenized_example["input_ids"] = input_ids
                tokenized_example["attention_mask"] = tokenized_examples["attention_mask"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                start_char = answers["start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_example["start_positions"].append(cls_index)
                    tokenized_example["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_example["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_example["end_positions"].append(token_end_index + 1)
                    # print(self.tokenizer.decode(tokenized_examples["input_ids"][i][token_start_index-1: token_end_index+2]))
                    # print(answers["text"])
                self.inputs.append(tokenized_example)

     # Validation preprocessing    
    def prepare_validation_features(self, relevant=None):
        self.inputs = []
        for idx, examples in enumerate(self.data):
            self.data[idx]["index"] = idx
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples['question'] = examples['question'].lstrip()
            if relevant is None:
                context = self.context[examples["relevant"]]
            else:
                context = self.context[examples["paragraphs"][relevant[idx]]]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = self.tokenizer(
                examples["question"],
                context,
                truncation="only_second",
                max_length=self.max_len,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.

            for i in range(len(tokenized_examples["input_ids"])):
                tokenized_example = {}
                tokenized_example["example_id"] = []
                tokenized_example["input_ids"] = tokenized_examples["input_ids"][i]
                tokenized_example["attention_mask"] = tokenized_examples["attention_mask"][i]

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1

                # One example can give several spans, this is the index of the example containing this span of text.
                # sample_index = sample_mapping
                # tokenized_example["example_id"].append(idx)
                tokenized_example["example_id"].append(examples["id"])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_example["offset_mapping"] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
                self.inputs.append(tokenized_example)

    def __len__(self):
        if self.preprocess:
            return len(self.inputs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.preprocess:
            return self.inputs[index]
        else:
            return self.data[index]

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
            sentences = [self.context[p] for p in examples["paragraphs"]]

            tokenized_example = self.tokenizer(question_headers, sentences, padding=True, truncation=True, max_length=self.max_len)
            self.tokenized_examples.append({k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_example.items()})
            if split != 'test':
                self.tokenized_examples[-1]["labels"] = examples["paragraphs"].index(examples["relevant"])

    def __len__(self) -> int:
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]
