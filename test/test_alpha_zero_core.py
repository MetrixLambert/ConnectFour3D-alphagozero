from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agents.alpha_zero_agent import AlphaZeroAgent
from environments.connect_four_3d import ConnectFour3D
from models.policy_value_network import PolicyValueNetwork
from mcts.mcts import MCTSNode
from self_play.replay_buffer import ReplayBuffer, SelfPlaySample
from training.trainer import AlphaZeroTrainer
from run_training import load_training_args
from run_evaluation import load_evaluation_args
from train import get_snapshot_path
from evaluation.checkpoint_compare import format_comparison_table
from utils.symmetry import apply_horizontal_symmetry, augment_batch_with_symmetry
from run_stage2_training import load_stage2_args
from ui.connect_four_ui import ConnectFourUI


def make_small_agent(simulations=2):
    env = ConnectFour3D()
    network = PolicyValueNetwork(channels=8, num_residual_blocks=1)
    agent = AlphaZeroAgent(
        env.observation_space,
        env.action_space,
        env,
        network,
        num_simulations=simulations,
        device="cpu",
        seed=0,
    )
    return env, network, agent


def test_ui_undo_restores_one_complete_position_snapshot():
    ui = ConnectFourUI.__new__(ConnectFourUI)
    ui.env = ConnectFour3D()
    ui.observation, _ = ui.env.reset()
    ui.human_player = 1
    ui.terminated = False
    ui.move_count = 0
    ui.moves = []
    ui.position_history = []
    ui.current_value = 0.25
    ui.value_history = [
        {"step": 0, "player": 1, "value": 0.25, "agent_value": -0.25}
    ]
    ui.suggested_action = None
    ui.suggested_position = None
    ui.calculating = False
    ui.game_id = 1
    ui.agent = SimpleNamespace(reset_episode=lambda: None)
    ui.suggestion_var = SimpleNamespace(set=lambda _value: None)
    ui.value_var = SimpleNamespace(set=lambda _value: None)
    ui._refresh = lambda *_args, **_kwargs: None
    ui._draw_value_plot = lambda: None
    ui.request_agent_suggestion = lambda: None
    ui.request_value_estimate = lambda: None

    ui._apply_action(0)
    assert ui.move_count == 1
    assert ui.env.board[0, 0, 0] == 1
    assert len(ui.position_history) == 1

    ui.undo_move()
    assert ui.move_count == 0
    assert np.count_nonzero(ui.env.board) == 0
    assert ui.env.current_player == 1
    assert ui.env.winner is None
    assert ui.moves == []
    assert ui.value_history[0]["value"] == 0.25
    assert ui.position_history == []


def test_policy_value_network_shapes():
    network = PolicyValueNetwork(channels=8, num_residual_blocks=1)
    logits, values = network(torch.zeros(3, 1, 5, 5, 5))
    assert logits.shape == (3, 25)
    assert values.shape == (3,)
    assert torch.all(values >= -1) and torch.all(values <= 1)


def test_mcts_returns_a_legal_normalized_policy():
    env, _, agent = make_small_agent()
    observation, _ = env.reset()
    action = agent.select_action(observation, training=True)
    policy = agent.get_policy()

    assert action in env.get_valid_actions()
    assert policy is not None
    assert policy.shape == (25,)
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy >= 0.0)


def test_temperature_schedule_changes_sampling_not_policy_target():
    env, _, agent = make_small_agent()
    agent.temperature_drop_move = 1
    agent.post_drop_temperature = 0.0
    root = MCTSNode(to_play=1, visit_count=4)
    root.children = {
        0: MCTSNode(to_play=-1, visit_count=3),
        1: MCTSNode(to_play=-1, visit_count=1),
    }
    agent.mcts.search = lambda *_args, **_kwargs: root
    observation, _ = env.reset()

    agent.select_action(observation, training=True)
    second_action = agent.select_action(observation, training=True)

    assert second_action == 0
    assert np.allclose(agent.get_policy()[:2], [0.75, 0.25])
    assert agent.episode_move_count == 2
    agent.reset_episode()
    assert agent.episode_move_count == 0


def test_late_game_simulation_budget_only_changes_training_search():
    env = ConnectFour3D()
    network = PolicyValueNetwork(channels=8, num_residual_blocks=1)
    agent = AlphaZeroAgent(
        env.observation_space,
        env.action_space,
        env,
        network,
        num_simulations=2,
        late_game_simulations=5,
        late_game_simulation_move=2,
        device="cpu",
        seed=0,
    )
    root = MCTSNode(to_play=1, visit_count=1)
    root.children = {0: MCTSNode(to_play=-1, visit_count=1)}
    used_budgets = []

    def fake_search(*_args, **_kwargs):
        used_budgets.append(agent.mcts.num_simulations)
        return root

    agent.mcts.search = fake_search
    observation, _ = env.reset()
    agent.select_action(observation, training=True)
    agent.select_action(observation, training=True)
    agent.select_action(observation, training=True)
    agent.select_action(observation, training=False)

    assert used_budgets == [2, 2, 5, 2]


def test_replay_buffer_samples_fixed_ply_stage_quotas():
    buffer = ReplayBuffer(capacity=400, seed=3)
    policy = np.full(25, 1.0 / 25, dtype=np.float32)
    samples = []
    for _ in range(100):
        for ply in (5, 15, 25, 35):
            state = np.zeros((5, 5, 5), dtype=np.float32)
            state.reshape(-1)[:ply] = 1.0
            samples.append(SelfPlaySample(state, policy, 1.0))
    buffer.add_game(samples)

    states, _, _ = buffer.sample(
        40,
        recent_window=200,
        recent_fraction=0.5,
        ply_fractions=(0.35, 0.30, 0.25, 0.10),
    )
    plies = np.count_nonzero(states, axis=(1, 2, 3))

    assert np.count_nonzero(plies < 10) == 14
    assert np.count_nonzero((plies >= 10) & (plies < 20)) == 12
    assert np.count_nonzero((plies >= 20) & (plies < 30)) == 10
    assert np.count_nonzero(plies >= 30) == 4
    assert len({state.tobytes() for state in states}) == 4  # four stage templates


def test_training_step_updates_from_buffer():
    _, network, _ = make_small_agent()
    buffer = ReplayBuffer(seed=0)
    uniform_policy = np.full(25, 1.0 / 25, dtype=np.float32)
    buffer.add_game(
        [
            SelfPlaySample(
                state=np.zeros((5, 5, 5), dtype=np.float32),
                policy=uniform_policy,
                value=0.0,
            )
            for _ in range(4)
        ]
    )
    trainer = AlphaZeroTrainer(network, buffer, device="cpu")
    losses = trainer.train_steps(steps=1, batch_size=2)
    assert losses["loss"] > 0.0


def test_checkpoint_restores_replay_buffer_and_progress(tmp_path):
    _, network, _ = make_small_agent()
    buffer = ReplayBuffer(capacity=7, seed=123)
    policies = np.eye(25, dtype=np.float32)
    samples = [
        SelfPlaySample(
            state=np.full((5, 5, 5), index, dtype=np.float32),
            policy=policies[index],
            value=float(index - 1),
        )
        for index in range(3)
    ]
    buffer.add_game(samples)
    trainer = AlphaZeroTrainer(network, buffer, device="cpu")
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(
        checkpoint_path,
        metadata={"iteration": 4, "total_games": 12},
    )

    _, restored_network, _ = make_small_agent()
    restored_buffer = ReplayBuffer(capacity=1, seed=999)
    restored_trainer = AlphaZeroTrainer(
        restored_network, restored_buffer, device="cpu"
    )
    metadata = restored_trainer.load_checkpoint(
        checkpoint_path,
        learning_rate=3e-4,
    )

    assert metadata == {"iteration": 4, "total_games": 12}
    assert restored_trainer.optimizer.param_groups[0]["lr"] == 3e-4
    assert len(restored_buffer) == 3
    assert restored_buffer._samples.maxlen == 7
    restored = list(restored_buffer._samples)
    for expected, actual in zip(samples, restored):
        assert np.array_equal(actual.state, expected.state)
        assert np.array_equal(actual.policy, expected.policy)
        assert actual.value == expected.value

    # Sampling must continue from the exact saved RNG state.
    expected_batch = buffer.sample(2)
    actual_batch = restored_buffer.sample(2)
    for expected, actual in zip(expected_batch, actual_batch):
        assert np.array_equal(actual, expected)


def test_checkpoint_can_expand_replay_buffer_on_resume(tmp_path):
    _, network, _ = make_small_agent()
    buffer = ReplayBuffer(capacity=3, seed=1)
    buffer.add_game(
        [
            SelfPlaySample(
                state=np.full((5, 5, 5), index, dtype=np.float32),
                policy=np.full(25, 1.0 / 25, dtype=np.float32),
                value=0.0,
            )
            for index in range(3)
        ]
    )
    path = tmp_path / "small_buffer.pt"
    AlphaZeroTrainer(network, buffer, device="cpu").save_checkpoint(path)

    _, restored_network, _ = make_small_agent()
    restored_buffer = ReplayBuffer(capacity=1, seed=2)
    AlphaZeroTrainer(
        restored_network, restored_buffer, device="cpu"
    ).load_checkpoint(path, replay_buffer_capacity=8)

    assert len(restored_buffer) == 3
    assert restored_buffer._samples.maxlen == 8


def test_replay_buffer_stratifies_recent_and_historical_samples():
    buffer = ReplayBuffer(capacity=20, seed=5)
    buffer.add_game(
        [
            SelfPlaySample(
                state=np.full((5, 5, 5), index, dtype=np.float32),
                policy=np.full(25, 1.0 / 25, dtype=np.float32),
                value=float(index),
            )
            for index in range(20)
        ]
    )

    states, _, values = buffer.sample(
        batch_size=10,
        recent_window=5,
        recent_fraction=0.5,
    )

    assert np.count_nonzero(values >= 15) == 5
    assert np.count_nonzero(values < 15) == 5
    assert len(np.unique(values)) == 10
    assert np.array_equal(states[:, 0, 0, 0], values)


def test_replay_buffer_rejects_invalid_stratified_sampling_arguments():
    buffer = ReplayBuffer(capacity=2, seed=0)
    buffer.add_game(
        [
            SelfPlaySample(
                state=np.zeros((5, 5, 5), dtype=np.float32),
                policy=np.full(25, 1.0 / 25, dtype=np.float32),
                value=0.0,
            )
        ]
    )

    with np.testing.assert_raises(ValueError):
        buffer.sample(1, recent_window=0, recent_fraction=0.5)
    with np.testing.assert_raises(ValueError):
        buffer.sample(1, recent_window=1, recent_fraction=1.1)


def test_json_training_config_converts_to_cli_arguments(tmp_path):
    config_path = tmp_path / "train.json"
    config_path.write_text(
        '{"iterations": 3, "learning_rate": 0.01, '
        '"resume": true, "unused": null}',
        encoding="utf-8",
    )

    assert load_training_args(config_path) == [
        "--iterations",
        "3",
        "--learning-rate",
        "0.01",
        "--resume",
    ]


def test_json_evaluation_config_converts_to_cli_arguments(tmp_path):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        '{"checkpoint": "model.pt", "games": 12, '
        '"alpha_beta_depth": 3, "device": "cpu"}',
        encoding="utf-8",
    )

    assert load_evaluation_args(config_path) == [
        "model.pt",
        "--games",
        "12",
        "--alpha-beta-depth",
        "3",
        "--device",
        "cpu",
    ]


def test_numbered_snapshot_path():
    assert get_snapshot_path(
        checkpoint=Path("saved_models/stage1.pt"),
        snapshot_dir=Path("saved_models/checkpoints"),
        iteration=20,
    ) == Path("saved_models/checkpoints/stage1_iter_000020.pt")


def test_json_checkpoint_comparison_config(tmp_path):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        '{"checkpoints": ["a.pt", "b.pt"], "games": 20}',
        encoding="utf-8",
    )

    assert load_evaluation_args(config_path) == [
        "--checkpoints",
        "a.pt",
        "b.pt",
        "--games",
        "20",
    ]


def test_json_checkpoint_arena_config(tmp_path):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        '{"checkpoints": ["a.pt", "b.pt"], '
        '"opponent_checkpoint": "champion.pt", "games": 20}',
        encoding="utf-8",
    )

    assert load_evaluation_args(config_path) == [
        "--checkpoints", "a.pt", "b.pt",
        "--opponent-checkpoint", "champion.pt",
        "--games", "20",
    ]


def test_json_selects_batch_opponents(tmp_path):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        '{"checkpoints": ["candidate.pt"], '
        '"opponent_checkpoint": "champion.pt", '
        '"opponents": ["checkpoint"]}',
        encoding="utf-8",
    )

    assert load_evaluation_args(config_path) == [
        "--checkpoints", "candidate.pt",
        "--opponent-checkpoint", "champion.pt",
        "--opponents", "checkpoint",
    ]


def test_checkpoint_comparison_table():
    table = format_comparison_table(
        [
            {
                "checkpoint": "models/iter_10.pt",
                "iteration": 10,
                "games": 100,
                "wins": 12,
                "draws": 3,
                "losses": 5,
            }
        ]
    )
    assert "iter_10.pt" in table
    assert "60.0%" in table


def test_all_horizontal_symmetries_keep_policy_aligned_with_state():
    state = np.zeros((5, 5, 5), dtype=np.float32)
    state[0, 1, 3] = 1.0
    policy = np.zeros(25, dtype=np.float32)
    policy[5 * 0 + 1] = 1.0
    transformed_positions = set()

    for symmetry in range(8):
        transformed_state, transformed_policy = apply_horizontal_symmetry(
            state, policy, symmetry
        )
        x, y, z = np.argwhere(transformed_state == 1.0)[0]
        action = int(np.argmax(transformed_policy))

        assert (x, y) == (action // 5, action % 5)
        assert z == 3
        assert np.isclose(transformed_policy.sum(), 1.0)
        assert transformed_state.flags.c_contiguous
        assert transformed_policy.flags.c_contiguous
        transformed_positions.add((int(x), int(y)))

    assert len(transformed_positions) == 8


def test_batch_symmetry_augmentation_preserves_shapes_and_policy_mass():
    states = np.zeros((4, 5, 5, 5), dtype=np.float32)
    policies = np.full((4, 25), 1.0 / 25, dtype=np.float32)
    augmented_states, augmented_policies = augment_batch_with_symmetry(
        states, policies, np.random.default_rng(0)
    )

    assert augmented_states.shape == states.shape
    assert augmented_policies.shape == policies.shape
    assert np.allclose(augmented_policies.sum(axis=1), 1.0)


def test_stage2_partially_reuses_stage1_weights_and_buffer(tmp_path):
    source_network = PolicyValueNetwork(
        channels=8,
        num_residual_blocks=1,
        value_head_channels=1,
        value_hidden_size=8,
    )
    with torch.no_grad():
        source_network.stem[0].weight.fill_(0.125)
        source_network.policy_head[-1].bias.fill_(0.25)
    source_buffer = ReplayBuffer(capacity=7, seed=4)
    source_buffer.add_game(
        [
            SelfPlaySample(
                state=np.zeros((5, 5, 5), dtype=np.float32),
                policy=np.full(25, 1.0 / 25, dtype=np.float32),
                value=1.0,
            )
        ]
    )
    source_trainer = AlphaZeroTrainer(source_network, source_buffer, device="cpu")
    source_path = tmp_path / "stage1.pt"
    source_trainer.save_checkpoint(
        source_path,
        metadata={"iteration": 230, "total_games": 2300},
    )

    target_network = PolicyValueNetwork(
        channels=8,
        num_residual_blocks=3,
        value_head_channels=4,
        value_hidden_size=32,
    )
    target_buffer = ReplayBuffer(capacity=1, seed=9)
    target_trainer = AlphaZeroTrainer(target_network, target_buffer, device="cpu")
    metadata, report = target_trainer.initialize_from_checkpoint(source_path)

    assert metadata["iteration"] == 230
    assert len(target_buffer) == 1
    assert target_buffer._samples.maxlen == 7
    assert torch.all(target_network.stem[0].weight == 0.125)
    assert torch.all(target_network.policy_head[-1].bias == 0.25)
    assert torch.count_nonzero(
        target_network.residual_tower[1].layers[-1].weight
    ) == 0
    assert any(name.startswith("value_head.") for name in report["skipped"])
    assert any(
        name.startswith("residual_tower.1.") for name in report["missing"]
    )


def test_expanded_value_head_shapes():
    network = PolicyValueNetwork(
        channels=8,
        num_residual_blocks=2,
        value_head_channels=4,
        value_hidden_size=32,
    )
    logits, values = network(torch.zeros(2, 1, 5, 5, 5))

    assert logits.shape == (2, 25)
    assert values.shape == (2,)
    assert network.value_head[0].in_features == 4 * 5**3
    assert network.value_head[0].out_features == 32


def test_deeper_network_reuses_all_old_layers_and_preserves_outputs(tmp_path):
    source = PolicyValueNetwork(
        channels=8,
        num_residual_blocks=2,
        value_head_channels=4,
        value_hidden_size=32,
    )
    source.eval()
    source_buffer = ReplayBuffer(capacity=3, seed=1)
    source_path = tmp_path / "source.pt"
    AlphaZeroTrainer(source, source_buffer, device="cpu").save_checkpoint(
        source_path,
        metadata={"iteration": 620, "total_games": 6200},
    )

    target = PolicyValueNetwork(
        channels=8,
        num_residual_blocks=4,
        value_head_channels=4,
        value_hidden_size=32,
    )
    target_buffer = ReplayBuffer(capacity=3, seed=2)
    trainer = AlphaZeroTrainer(target, target_buffer, device="cpu")
    _, report = trainer.initialize_from_checkpoint(
        source_path,
        excluded_prefixes=(),
    )

    inputs = torch.randn(3, 1, 5, 5, 5)
    target.eval()
    with torch.no_grad():
        source_policy, source_value = source(inputs)
        target_policy, target_value = target(inputs)

    assert report["skipped"] == []
    assert all(
        name.startswith(("residual_tower.2.", "residual_tower.3."))
        for name in report["missing"]
    )
    assert torch.allclose(target_policy, source_policy, atol=1e-6)
    assert torch.allclose(target_value, source_value, atol=1e-6)


def test_stage2_runner_switches_from_initialize_to_resume(tmp_path):
    config_path = tmp_path / "configs" / "train_stage2.json"
    config_path.parent.mkdir()
    config_path.write_text(
        '{"iterations": 2, "checkpoint": "models/stage2.pt", '
        '"initialize_from": "stage1.pt", "pretraining_steps": 5, '
        '"resume": false}',
        encoding="utf-8",
    )

    initialize_args, is_resume = load_stage2_args(config_path)
    assert not is_resume
    assert "--initialize-from" in initialize_args
    assert "--pretraining-steps" in initialize_args

    checkpoint = tmp_path / "models" / "stage2.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    resume_args, is_resume = load_stage2_args(config_path)
    assert is_resume
    assert "--initialize-from" not in resume_args
    assert "--pretraining-steps" not in resume_args
    assert "--resume" in resume_args
