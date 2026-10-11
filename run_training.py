"""Run training from configs/train.json without command-line arguments."""

from __future__ import annotations

import json
from pathlib import Path

from train import main as train_main


CONFIG_PATH = Path(__file__).parent / "configs" / "train.json"


def load_training_args(path: Path = CONFIG_PATH) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")

    args: list[str] = []
    for name, value in config.items():
        option = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, list):
            args.append(option)
            args.extend(str(item) for item in value)
        elif value is not None:
            args.extend((option, str(value)))
    return args


if __name__ == "__main__":
    print(f"Loading training configuration from {CONFIG_PATH}", flush=True)
    train_main(load_training_args())
