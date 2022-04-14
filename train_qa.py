import os
import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator, 
    EvalPrediction
)
from dataset import QADataset

import torch
from utils import load_dataset

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese")

    # load dataset
    context, data = load_dataset(args)

    train_dataset = QADataset(context, data["train"], tokenizer, args.max_len)
    valid_dataset = QADataset(context, data["valid"], tokenizer, args.max_len)

    data_collator = DefaultDataCollator()

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
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
    parser.add_argument("--lr", type=float, default=3e-5)
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
