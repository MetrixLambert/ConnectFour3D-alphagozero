"""Run the 200-simulation stage2 branch starting from iteration 340."""

from pathlib import Path

from run_training import load_training_args
from train import main as train_main


CONFIG_PATH = (
    Path(__file__).parent / "configs" / "train_stage2_sim200_from340.json"
)


if __name__ == "__main__":
    print(
        f"Loading sim200 training configuration from {CONFIG_PATH}",
        flush=True,
    )
    train_main(load_training_args(CONFIG_PATH))
