"""Fine-tune the iteration-480 champion with LR 1e-4."""

from pathlib import Path

from run_training import load_training_args
from train import main as train_main


CONFIG_PATH = (
    Path(__file__).parent
    / "configs"
    / "train_stage2_sim200_lr1e4_from480.json"
)


if __name__ == "__main__":
    print(
        f"Loading low-LR training configuration from {CONFIG_PATH}",
        flush=True,
    )
    train_main(load_training_args(CONFIG_PATH))
