"""Train from iteration 620 with 75% of each batch drawn from recent replay."""

from pathlib import Path

from run_training import load_training_args
from train import main as train_main


CONFIG_PATH = Path(__file__).parent / "configs" / "train_stage2_recent75_from620.json"


if __name__ == "__main__":
    print(
        f"Loading recent-75 training configuration from {CONFIG_PATH}",
        flush=True,
    )
    train_main(load_training_args(CONFIG_PATH))
