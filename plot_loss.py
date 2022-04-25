import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/qa/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    state_path = os.path.join(args.ckpt_dir, 'trainer_state.json')
    with open(state_path) as f:
        data = json.load(f)

    log_data = data["log_history"]
    loss_data = []
    for log in log_data[:-2]:
        if "loss" in log.keys():
            loss_data.append(log["loss"])
    print(len(loss_data))
    plt.plot(
        list(range(0, len(loss_data)*500, 500)),
        loss_data,
        color="orange",
        label="train",
    )
    plt.title("learning curve")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('loss_curve.png')
