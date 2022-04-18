import csv
import os
import json
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForMultipleChoice,
    TrainingArguments, 
    Trainer, 
)
from dataset import QADataset, contextDataset
from collate_fn import DataCollatorForMultipleChoice, DataCollatorForQuestionAnswering
import torch
from utils_qa import postprocess_qa_predictions

question_column_name = "question"
answer_column_name = "answer"

def load_dataset(args):
    context_data_path = os.path.join(args.context_dir)
    with  open(context_data_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    
    data = {}
    data_paths = os.path.join(args.test_dir)
    with  open(data_paths, 'r', encoding="utf-8") as f:
        data["test"] = json.load(f)
    return context, data

def main(args):
    # load dataset
    context, data = load_dataset(args)

    context_tokenizer = AutoTokenizer.from_pretrained(args.context_ckpt_dir)
    context_model = AutoModelForMultipleChoice.from_pretrained(args.context_ckpt_dir)
    test_dataset = contextDataset(context, data["test"], context_tokenizer, args.max_len, 'test')

    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_ckpt_dir)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_ckpt_dir)

    training_args = TrainingArguments(
        output_dir=args.result_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=args.weight_decay,
        do_predict=True
    )

    context_trainer = Trainer(
        model=context_model,
        args=training_args,
        tokenizer=context_tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=context_tokenizer, split="predict"),
    )

    qa_trainer = QuestionAnsweringTrainer(
        model=qa_model,
        args=training_args,
        tokenizer=qa_tokenizer,
        data_collator=DataCollatorForQuestionAnswering(),
    )

    if training_args.do_predict:
        results = context_trainer.predict(test_dataset)
        predictions = results.predictions
        preds = np.argmax(predictions, axis=1)
        print(preds)

        test_features = QADataset(context, data["test"], qa_tokenizer, args.max_len, 'test', relevant=preds)
        test_examples = QADataset(context, data["test"], qa_tokenizer, args.max_len, 'test', relevant=preds, preprocess=False)
        results = qa_trainer.predict(test_features)
        predictions = postprocess_qa_predictions(test_examples, test_features, results.predictions, qa_tokenizer)

        preds_list = []
        preds_list.append(['id', 'answer'])
        for id, ans in predictions.items():
            preds_list.append([id, ans])

        # print(predictions)
        with open(args.submission_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(preds_list)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--context_ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/context/checkpoint-7000/",
    )
    parser.add_argument(
        "--qa_ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/qa/",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Directory to the result.",
        default="./results/",
    )
    parser.add_argument(
        "--submission_csv",
        type=Path,
        default="submission.csv",
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
