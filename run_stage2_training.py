"""Initialize stage2 once, then resume it on subsequent runs."""

import json
from pathlib import Path

from run_training import load_training_args
from train import main as train_main


CONFIG_PATH = Path(__file__).parent / "configs" / "train_stage2.json"


def load_stage2_args(path: Path = CONFIG_PATH) -> tuple[list[str], bool]:
    args = load_training_args(path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    checkpoint = path.parent.parent / config["checkpoint"]
    is_resume = checkpoint.is_file()
    if is_resume:
        for option in ("--initialize-from", "--pretraining-steps"):
            if option in args:
                index = args.index(option)
                del args[index:index + 2]
        if "--resume" not in args:
            args.append("--resume")
    return args, is_resume


if __name__ == "__main__":
    training_args, resuming = load_stage2_args()
    mode = "resume" if resuming else "initialize"
    print(
        f"Loading stage2 configuration from {CONFIG_PATH}; mode={mode}",
        flush=True,
    )
    train_main(training_args)
