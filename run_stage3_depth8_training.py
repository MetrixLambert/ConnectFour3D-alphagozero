"""Initialize the 8-block stage3 network once, then resume it."""

from pathlib import Path

from run_stage2_training import load_stage2_args
from train import main as train_main


CONFIG_PATH = Path(__file__).parent / "configs" / "train_stage3_depth8_from620.json"


if __name__ == "__main__":
    training_args, resuming = load_stage2_args(CONFIG_PATH)
    mode = "resume" if resuming else "initialize"
    print(
        f"Loading stage3 depth-8 configuration from {CONFIG_PATH}; mode={mode}",
        flush=True,
    )
    train_main(training_args)
