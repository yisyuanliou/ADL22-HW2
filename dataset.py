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
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            # for i, offset in enumerate(offset_mapping):
            answer = answers
            start_char = answer["start"][0]
            end_char = answer["start"][0] + len(answer["text"][0])
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
