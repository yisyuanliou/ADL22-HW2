import os
import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataset import contextDataset

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

TRAIN = "train"
VALID = "valid"
DEV = "test"
SPLITS = [TRAIN, VALID, DEV]

def load_dataset(args):
    context_data_path = os.path.join(args.data_dir, "context.json")
    with  open(context_data_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    
    data = {}
    data_paths = {split: os.path.join(args.data_dir, f"{split}.json") for split in SPLITS}
    for split in SPLITS:
        data_paths = os.path.join(args.data_dir, f"{split}.json")
        with  open(data_paths, 'r', encoding="utf-8") as f:
            data[split] = json.load(f)
    return context, data

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "labels"
        labels = [feature[label_name] for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"][0])
        flattened_features = [
            [{k: v[0][i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMultipleChoice.from_pretrained("bert-base-chinese")

    # load dataset
    context, data = load_dataset(args)

    train_dataset = contextDataset(context, data["train"], tokenizer, args.max_len, 'train')
    valid_dataset = contextDataset(context, data["valid"], tokenizer, args.max_len, 'valid')

    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=args.weight_decay,
        do_train=True
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        print(metrics)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        print(metrics)
        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/context/",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="model name.",
        default="model",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=float, default=3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=3)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
