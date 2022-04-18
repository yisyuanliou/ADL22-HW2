import os
import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    EvalPrediction
)
from collate_fn import DataCollatorForQuestionAnswering
from dataset import QADataset
from datasets import load_metric

import torch
from utils import load_dataset

question_column_name = "question"
answer_column_name = "answer"
        
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # load dataset
    context, data = load_dataset(args)

    train_dataset  = QADataset(context, data["train"], tokenizer, args.max_len, 'train')
    valid_dataset  = QADataset(context, data["valid"], tokenizer, args.max_len, 'train')
    valid_features = QADataset(context, data["valid"], tokenizer, args.max_len, 'valid')
    valid_example = QADataset(context, data["valid"], tokenizer, args.max_len, 'valid', preprocess=False)

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
    metric = load_metric("squad_v2")

    # Metric
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        eval_examples=valid_example,
        tokenizer=tokenizer,
        data_collator=DataCollatorForQuestionAnswering(),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            print(training_args.resume_from_checkpoint)
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        print(metrics)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(eval_dataset=valid_features)
    #     print(metrics)
    #     # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


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
        default="./ckpt/qa/",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="model name.",
        default="bert-base-chinese",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=float, default=2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
