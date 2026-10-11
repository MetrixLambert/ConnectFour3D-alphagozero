"""Run evaluation from configs/eval.json without command-line arguments."""

from __future__ import annotations

import json
from pathlib import Path

from eval import main as evaluation_main


CONFIG_PATH = Path(__file__).parent / "configs" / "eval.json"


def load_evaluation_args(path: Path = CONFIG_PATH) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must be a JSON object: {path}")
    has_checkpoint = "checkpoint" in config
    has_checkpoints = "checkpoints" in config
    if has_checkpoint == has_checkpoints:
        raise ValueError(
            f"Evaluation config requires exactly one of 'checkpoint' or "
            f"'checkpoints': {path}"
        )

    args = [str(config["checkpoint"])] if has_checkpoint else []
    for name, value in config.items():
        if name == "checkpoint" or value is None:
            continue
        option = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, list):
            args.append(option)
            args.extend(str(item) for item in value)
        else:
            args.extend((option, str(value)))
    return args


if __name__ == "__main__":
    print(f"Loading evaluation configuration from {CONFIG_PATH}", flush=True)
    evaluation_main(load_evaluation_args())
