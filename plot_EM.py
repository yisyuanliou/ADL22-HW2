import csv
import os
import json
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt

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
SPLIT = "eval"

def load_dataset(args):
    context_data_path = os.path.join(args.context_dir)
    with  open(context_data_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    
    data = {}
    data_paths = os.path.join(args.eval_dir)
    with  open(data_paths, 'r', encoding="utf-8") as f:
        data[SPLIT] = json.load(f)
    return context, data

def main(args):
    # load dataset
    context, data = load_dataset(args)

    context_tokenizer = AutoTokenizer.from_pretrained(args.context_ckpt_dir)
    context_model = AutoModelForMultipleChoice.from_pretrained(args.context_ckpt_dir)
    eval_dataset = contextDataset(context, data[SPLIT], context_tokenizer, args.max_len, SPLIT)

    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_ckpt_dir)

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
        data_collator=DataCollatorForMultipleChoice(tokenizer=context_tokenizer, split="train"),
    )

    if training_args.do_predict:
        results = context_trainer.predict(eval_dataset)
        predictions = results.predictions
        preds = np.argmax(predictions, axis=1)
        print(preds)

    eval_features = QADataset(context, data[SPLIT], qa_tokenizer, args.max_len, SPLIT, relevant=preds)
    eval_examples = QADataset(context, data[SPLIT], qa_tokenizer, args.max_len, SPLIT, relevant=preds, preprocess=False)

    EM_data = []
    for i in range(args.start_ckpt, args.end_ckpt, args.period):
        ckpt_path = os.path.join(args.qa_ckpt_dir, "checkpoint-"+str(i))
        qa_model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
        qa_trainer = QuestionAnsweringTrainer(
            model=qa_model,
            args=training_args,
            tokenizer=qa_tokenizer,
            data_collator=DataCollatorForQuestionAnswering(),
        )

        if training_args.do_predict:
            results = qa_trainer.predict(eval_features)
            predictions = postprocess_qa_predictions(eval_examples, eval_features, results.predictions, qa_tokenizer)

            preds_list = []
            preds_list.append(["id", "answer"])
            EM = 0
            for i, (id, ans) in enumerate(predictions.items()):
                preds_list.append([id, ans])
                if i < 5:
                    print(ans, eval_examples[i]["answer"]["text"])
                EM += (ans == eval_examples[i]["answer"]["text"])
        print("EM", EM / len(eval_examples))
        EM_data.append(EM / len(eval_examples))

    plt.plot(
        list(range(args.start_ckpt, args.end_ckpt, args.period)),
        EM_data,
        color="orange",
        label="valid",
    )
    plt.title("learning curve")
    plt.xlabel("steps")
    plt.ylabel("EM")
    plt.legend()
    plt.savefig("EM_curve_"+SPLIT+".png")
            
            

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/valid.json",
    )
    parser.add_argument(
        "--context_ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/context/albert2/",
    )
    parser.add_argument(
        "--qa_ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/qa/macbert/",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Directory to the result.",
        default="./results/",
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

    parser.add_argument("--start_ckpt", type=int, default=500)
    parser.add_argument("--end_ckpt", type=int, default=22000)
    parser.add_argument("--period", type=int, default=3000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
