"""Train from iteration 620 with a larger MCTS budget from ply 20 onward."""

from pathlib import Path

from run_training import load_training_args
from train import main as train_main


CONFIG_PATH = (
    Path(__file__).parent / "configs" / "train_stage2_dynamic_sim_from620.json"
)


if __name__ == "__main__":
    print(
        f"Loading dynamic-simulation training configuration from {CONFIG_PATH}",
        flush=True,
    )
    train_main(load_training_args(CONFIG_PATH))
