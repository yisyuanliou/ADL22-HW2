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
    def __init__(self, context, data, tokenizer, max_len):
        self.context = context
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess()

    def preprocess(self):
        self.inputs = []
        for examples in self.data:
            questions = examples["question"].strip()
            tokenized_example = self.tokenizer(
                questions,
                self.context[examples["relevant"]],
                max_length=self.max_len,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset = tokenized_example.pop("offset_mapping")
            answers = examples["answer"]
            start_positions = []
            end_positions = []

            # for i, offset in enumerate(offset_mapping):
            answer = answers
            start_char = answer["start"]
            end_char = answer["start"] + len(answer["text"])
            sequence_ids = tokenized_example.sequence_ids()

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

            tokenized_example["start_positions"] = start_positions
            tokenized_example["end_positions"] = end_positions
            self.inputs.append(tokenized_example)   


    # Training preprocessing
    def prepare_train_features(self, examples):
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

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answer'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["start"][0]
                end_char = start_char + len(answers["text"][0])

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
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

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
