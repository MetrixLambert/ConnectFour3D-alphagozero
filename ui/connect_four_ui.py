"""Local Tkinter UI for playing Connect Four 3D against selectable agents."""

from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Allow both ``python -m ui.connect_four_ui`` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.alpha_beta_agent import AlphaBetaAgent
from agents.alpha_zero_agent import AlphaZeroAgent
from environments.connect_four_3d import ConnectFour3D
from models.policy_value_network import PolicyValueNetwork


class ConnectFourUI:
    """Keep presentation state separate from the environment's game state."""

    PLAYER_COLORS = {1: "#e74c3c", -1: "#2878d0"}
    PLAYER_NAMES = {1: "Player 1（红）", -1: "Player 2（蓝）"}
    CELL_SIZE = 42
    MARGIN = 28
    ALPHA_ZERO = "AlphaZero（Stage3 冠军 iter 130）"
    ALPHA_ZERO_PREVIOUS = "AlphaZero（上一任 Stage3 冠军 iter 100）"
    ALPHA_ZERO_CUSTOM = "AlphaZero（自定义 checkpoint）"
    ALPHA_BETA = "Alpha-Beta"
    ALPHA_ZERO_PRESETS = {
        ALPHA_ZERO: "stage3_depth8_champion_iter_000130.pt",
        ALPHA_ZERO_PREVIOUS: "stage3_depth8_champion_iter_000100.pt",
    }
    SIMULATION_PRESETS = (50, 100, 200, 400)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Connect Four 3D")

        self.env = ConnectFour3D()
        self.agent: AlphaBetaAgent | AlphaZeroAgent | None = None
        self.agent_cache_key: tuple[object, ...] | None = None
        self.agent_details: dict[str, object] = {}
        self.agent_checkpoint_metadata: dict[str, object] = {}
        self.human_player = 1
        self.observation: np.ndarray | None = None
        self.terminated = False
        self.move_count = 0
        self.suggested_action: int | None = None
        self.suggested_position: tuple[int, int, int] | None = None
        self.calculating = False
        self.inference_lock = threading.Lock()
        self.game_id = 0
        self.game_started_at: datetime | None = None
        self.record_path: Path | None = None
        self.moves: list[dict[str, object]] = []
        self.position_history: list[dict[str, object]] = []
        self.current_value: float | None = None
        self.value_history: list[dict[str, float | int]] = []

        self.human_player_var = tk.IntVar(value=1)
        self.agent_type_var = tk.StringVar(value=self.ALPHA_ZERO)
        self.alpha_beta_depth = 3
        self.alpha_zero_simulations = 400
        self.alpha_zero_device = "auto"
        self.alpha_zero_checkpoint = str(
            Path(__file__).resolve().parents[1]
            / "saved_models"
            / "deployment"
            / "stage3_depth8_champion_iter_000130.pt"
        )
        self.status_var = tk.StringVar()
        self.suggestion_var = tk.StringVar(value="Agent 建议：等待对局开始")
        self.value_var = tk.StringVar(value="Value network：等待局面")

        self.canvases: list[tk.Canvas] = []
        self._build_ui()
        self.new_game()

    def _build_ui(self) -> None:
        controls = ttk.LabelFrame(self.root, text="对局设置", padding=8)
        controls.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(controls, text="真人执棋：").grid(row=0, column=0, padx=4)
        ttk.Radiobutton(
            controls, text="Player 1（红）", variable=self.human_player_var, value=1
        ).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(
            controls, text="Player 2（蓝）", variable=self.human_player_var, value=-1
        ).grid(row=0, column=2, padx=4)
        ttk.Label(controls, text="对手模型：").grid(
            row=0, column=3, padx=(16, 4)
        )
        model_selector = ttk.Combobox(
            controls,
            textvariable=self.agent_type_var,
            values=(
                self.ALPHA_ZERO,
                self.ALPHA_ZERO_PREVIOUS,
                self.ALPHA_ZERO_CUSTOM,
                self.ALPHA_BETA,
            ),
            state="readonly",
            width=24,
        )
        model_selector.grid(row=0, column=4, padx=4)
        model_selector.bind("<<ComboboxSelected>>", self._on_agent_selected)
        ttk.Button(
            controls, text="模型参数…", command=self.open_agent_settings
        ).grid(
            row=0, column=5, padx=4
        )
        ttk.Button(controls, text="新对局", command=self.new_game).grid(
            row=0, column=6, padx=(16, 4)
        )
        board_frame = ttk.Frame(self.root)
        board_frame.pack(padx=10, pady=4)
        canvas_size = self.MARGIN * 2 + self.CELL_SIZE * self.env.board_size
        for z in range(self.env.board_size):
            panel = ttk.LabelFrame(
                board_frame,
                text=f"z = {z}" + ("（最底层）" if z == 0 else ""),
                padding=3,
            )
            panel.grid(row=z // 3, column=z % 3, padx=5, pady=5)
            canvas = tk.Canvas(
                panel,
                width=canvas_size,
                height=canvas_size,
                background="#f8f9fa",
                highlightthickness=0,
            )
            canvas.pack()
            # Every layer cell maps to the same vertical column (x, y).
            canvas.bind("<Button-1>", lambda event, layer=z: self._on_board_click(event, layer))
            self.canvases.append(canvas)

        value_panel = ttk.LabelFrame(board_frame, text="Value-step（Agent 视角）", padding=3)
        value_panel.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.value_figure = Figure(figsize=(3.05, 2.65), dpi=100)
        self.value_axis = self.value_figure.add_subplot(111)
        self.value_plot_canvas = FigureCanvasTkAgg(self.value_figure, master=value_panel)
        self.value_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_value_plot()

        info = ttk.LabelFrame(self.root, text="对局状态", padding=8)
        info.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Label(info, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.suggestion_var).pack(anchor="w", pady=(4, 6))
        ttk.Label(info, textvariable=self.value_var).pack(anchor="w", pady=(0, 6))

        buttons = ttk.Frame(info)
        buttons.pack(anchor="w")
        self.calculate_button = ttk.Button(
            buttons, text="AI 推荐当前方", command=self.request_agent_suggestion
        )
        self.calculate_button.pack(side="left", padx=(0, 6))
        self.confirm_button = ttk.Button(
            buttons, text="确认 Agent 落子", command=self.confirm_agent_move
        )
        self.confirm_button.pack(side="left")
        self.undo_button = ttk.Button(
            buttons, text="悔棋一步", command=self.undo_move
        )
        self.undo_button.pack(side="left", padx=(6, 0))
        ttk.Button(
            buttons, text="保存棋局记录", command=self.save_game_record
        ).pack(side="left", padx=(6, 0))

    def _on_agent_selected(self, _event: tk.Event | None = None) -> None:
        """Apply a bundled checkpoint preset before opening its settings."""
        preset = self.ALPHA_ZERO_PRESETS.get(self.agent_type_var.get())
        if preset is not None:
            self.alpha_zero_checkpoint = str(
                Path(__file__).resolve().parents[1]
                / "saved_models"
                / "deployment"
                / preset
            )
        self.open_agent_settings()

    def open_agent_settings(self) -> None:
        """Show the parameter dialog for the currently selected opponent."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{self.agent_type_var.get()} 参数")
        dialog.transient(self.root)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)

        if self.agent_type_var.get() == self.ALPHA_BETA:
            depth_var = tk.StringVar(value=str(self.alpha_beta_depth))
            ttk.Label(body, text="搜索深度 max_depth：").grid(
                row=0, column=0, sticky="w", padx=4, pady=6
            )
            ttk.Entry(body, textvariable=depth_var, width=12).grid(
                row=0, column=1, sticky="ew", padx=4, pady=6
            )

            def save() -> None:
                try:
                    depth = int(depth_var.get())
                    if depth <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("参数错误", "max_depth 必须是正整数。", parent=dialog)
                    return
                self.alpha_beta_depth = depth
                dialog.destroy()

        else:
            simulations_var = tk.StringVar(value=str(self.alpha_zero_simulations))
            device_var = tk.StringVar(value=self.alpha_zero_device)
            checkpoint_var = tk.StringVar(value=self.alpha_zero_checkpoint)
            ttk.Label(body, text="MCTS simulations：").grid(
                row=0, column=0, sticky="w", padx=4, pady=6
            )
            ttk.Combobox(
                body,
                textvariable=simulations_var,
                values=tuple(str(value) for value in self.SIMULATION_PRESETS),
                state="readonly",
                width=10,
            ).grid(row=0, column=1, sticky="w", padx=4, pady=6)
            ttk.Label(body, text="计算设备：").grid(
                row=1, column=0, sticky="w", padx=4, pady=6
            )
            ttk.Combobox(
                body,
                textvariable=device_var,
                values=("auto", "cuda", "cpu"),
                state="readonly",
                width=10,
            ).grid(row=1, column=1, sticky="w", padx=4, pady=6)
            ttk.Label(body, text="Checkpoint：").grid(
                row=2, column=0, sticky="w", padx=4, pady=6
            )
            ttk.Entry(body, textvariable=checkpoint_var, width=62).grid(
                row=2, column=1, sticky="ew", padx=4, pady=6
            )

            def browse() -> None:
                selected = filedialog.askopenfilename(
                    parent=dialog,
                    title="选择 AlphaZero checkpoint",
                    filetypes=(("PyTorch checkpoint", "*.pt"), ("所有文件", "*.*")),
                )
                if selected:
                    checkpoint_var.set(selected)

            ttk.Button(body, text="浏览…", command=browse).grid(
                row=2, column=2, padx=4, pady=6
            )

            def save() -> None:
                try:
                    simulations = int(simulations_var.get())
                    if simulations <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror(
                        "参数错误", "simulations 必须是正整数。", parent=dialog
                    )
                    return
                checkpoint_path = Path(checkpoint_var.get()).expanduser()
                if not checkpoint_path.is_file():
                    messagebox.showerror(
                        "参数错误", f"找不到 checkpoint：\n{checkpoint_path}", parent=dialog
                    )
                    return
                if device_var.get() == "cuda" and not torch.cuda.is_available():
                    messagebox.showerror("参数错误", "当前环境无法使用 CUDA。", parent=dialog)
                    return
                self.alpha_zero_simulations = simulations
                self.alpha_zero_device = device_var.get()
                self.alpha_zero_checkpoint = str(checkpoint_path.resolve())
                dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(
            side="right", padx=(6, 0)
        )
        ttk.Button(buttons, text="确定", command=save).pack(side="right")
        body.columnconfigure(1, weight=1)
        dialog.bind("<Return>", lambda _event: save())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()

    def _create_agent(self) -> AlphaBetaAgent | AlphaZeroAgent:
        """Create or reuse the agent selected in the controls."""
        if self.agent_type_var.get() == self.ALPHA_BETA:
            cache_key = (self.ALPHA_BETA, self.alpha_beta_depth)
            if self.agent is None or cache_key != self.agent_cache_key:
                self.agent = AlphaBetaAgent(
                    self.env.observation_space,
                    self.env.action_space,
                    self.env,
                    max_depth=self.alpha_beta_depth,
                )
                self.agent_cache_key = cache_key
            self.agent_details = {
                "agent": "AlphaBetaAgent",
                "max_depth": self.alpha_beta_depth,
            }
            return self.agent

        checkpoint_path = Path(self.alpha_zero_checkpoint).expanduser().resolve()
        device_name = self.alpha_zero_device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境无法使用 CUDA。")
        cache_key = (
            self.ALPHA_ZERO,
            str(checkpoint_path),
            self.alpha_zero_simulations,
            device_name,
        )
        if self.agent is None or cache_key != self.agent_cache_key:
            checkpoint = torch.load(
                checkpoint_path, map_location=device_name, weights_only=True
            )
            metadata = checkpoint.get("metadata", {})
            network = PolicyValueNetwork(
                board_size=self.env.board_size,
                num_actions=self.env.action_space.n,
                channels=int(metadata.get("channels", 32)),
                num_residual_blocks=int(metadata.get("residual_blocks", 2)),
                value_head_channels=int(metadata.get("value_head_channels", 1)),
                value_hidden_size=int(
                    metadata.get("value_hidden_size", metadata.get("channels", 32))
                ),
            )
            network.load_state_dict(checkpoint["model_state_dict"])
            network.eval()
            self.agent = AlphaZeroAgent(
                self.env.observation_space,
                self.env.action_space,
                env=self.env,
                network=network,
                num_simulations=self.alpha_zero_simulations,
                device=device_name,
            )
            self.agent_cache_key = cache_key
            self.agent_checkpoint_metadata = dict(metadata)
        metadata = self.agent_checkpoint_metadata
        self.agent_details = {
            "agent": "AlphaZeroAgent",
            "checkpoint": str(checkpoint_path),
            "iteration": int(metadata.get("iteration", -1)),
            "simulations": self.alpha_zero_simulations,
            "device": device_name,
        }
        return self.agent

    def new_game(self) -> None:
        """Create the selected agent and start a fresh game."""
        try:
            self.agent = self._create_agent()
        except Exception as exc:
            messagebox.showerror("Agent 加载失败", str(exc))
            return

        self.game_id += 1  # Invalidates results from an older background search.
        self.human_player = self.human_player_var.get()
        self.agent.reset_episode()
        self.observation, _ = self.env.reset()
        self.terminated = False
        self.move_count = 0
        self.suggested_action = None
        self.suggested_position = None
        self.calculating = False
        self.game_started_at = datetime.now().astimezone()
        filename = self.game_started_at.strftime("connect_four_3d_%Y%m%d_%H%M%S_%f.json")
        self.record_path = (
            Path(__file__).resolve().parents[1] / "records" / "games" / filename
        )
        self.moves = []
        self.position_history = []
        self.current_value = None
        self.value_history = []
        self.suggestion_var.set("Agent 建议：暂无")
        self.value_var.set("Value network：等待计算")
        self._draw_value_plot()
        self._refresh()
        if self.env.current_player != self.human_player:
            self.request_agent_suggestion()
        else:
            self.request_value_estimate()

    def _on_board_click(self, event: tk.Event, _layer: int) -> None:
        """Convert a clicked grid cell to the environment-defined action."""
        if self.terminated or self.calculating or self.env.current_player != self.human_player:
            return
        y = (event.x - self.MARGIN) // self.CELL_SIZE
        x = (event.y - self.MARGIN) // self.CELL_SIZE
        if not (0 <= x < self.env.board_size and 0 <= y < self.env.board_size):
            return
        action = self.env.board_size * int(x) + int(y)
        if not self.env._is_valid_action(action):
            messagebox.showinfo("无效落子", f"列 (x={x}, y={y}) 已满。")
            return
        self._apply_action(action)

    def _apply_action(self, action: int) -> None:
        """Apply every official move through env.step; never mutate env.board here."""
        player = int(self.env.current_player)
        x, y = self.env._action_to_position(action)
        z = self.env._get_drop_height(x, y)
        snapshot = {
            "board": self.env.board.copy(),
            "current_player": int(self.env.current_player),
            "winner": self.env.winner,
            "winning_positions": list(self.env.winning_positions),
            "terminated": bool(self.terminated),
            "move_count": int(self.move_count),
            "moves": list(self.moves),
            "current_value": self.current_value,
            "value_history": [dict(point) for point in self.value_history],
        }
        try:
            observation, _, terminated, _, info = self.env.step(action)
        except ValueError as exc:
            messagebox.showerror("落子失败", str(exc))
            return

        self.position_history.append(snapshot)
        self.observation = observation
        self.terminated = bool(terminated)
        self.move_count += 1
        self.moves.append(
            {
                "move": self.move_count,
                "player": player,
                "action": int(action),
                "position": {"x": int(x), "y": int(y), "z": int(z)},
                "played_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        self.suggested_action = None
        self.suggested_position = None
        self.suggestion_var.set("Agent 建议：暂无")
        self._refresh(info.get("winner"))
        if not self.terminated and self.env.current_player != self.human_player:
            self.request_agent_suggestion()
        elif not self.terminated:
            self.request_value_estimate()

    def request_agent_suggestion(self) -> None:
        """Run the selected agent off the Tk main thread and return its result."""
        if (
            self.terminated
            or self.calculating
            or self.agent is None
            or self.observation is None
        ):
            return

        self.calculating = True
        self.suggested_action = None
        self.suggested_position = None
        self.suggestion_var.set("Agent 正在计算…")
        self._refresh()
        game_id = self.game_id
        observation = self.observation.copy()
        agent = self.agent
        player = int(self.env.current_player)

        def worker() -> None:
            try:
                with self.inference_lock:
                    value = (
                        agent.evaluate_state(observation)
                        if isinstance(agent, AlphaZeroAgent)
                        else None
                    )
                    action = agent.select_action(observation, training=False)
                error = None
            except Exception as exc:  # Return failure to Tk instead of touching widgets.
                action = None
                value = None
                error = str(exc)
            self.root.after(
                0,
                self._finish_agent_suggestion,
                game_id,
                observation,
                player,
                action,
                value,
                error,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_agent_suggestion(
        self,
        game_id: int,
        searched_observation: np.ndarray,
        player: int,
        action: int | None,
        value: float | None,
        error: str | None,
    ) -> None:
        # Ignore stale work after a new game or any unexpected state change.
        if game_id != self.game_id or not np.array_equal(searched_observation, self.env.board):
            return
        self.calculating = False
        if value is not None:
            self._record_value_estimate(self.move_count, player, value)
        if error is not None or action is None:
            self.suggestion_var.set(f"Agent 计算失败：{error or '未返回动作'}")
            self._refresh()
            return

        x, y = self.env._action_to_position(action)
        z = self.env._get_drop_height(x, y)
        if z is None:
            self.suggestion_var.set("Agent 返回了已满列，请重新计算。")
            self._refresh()
            return
        self.suggested_action = int(action)
        self.suggested_position = (int(x), int(y), int(z))
        if player == self.human_player:
            instruction = "这是给真人的推荐，请点击对应棋盘列完成落子。"
        else:
            instruction = "请在实体棋盘落子后点击“确认 Agent 落子”。"
        self.suggestion_var.set(
            f"AI 推荐：action={action}，(x={x}, y={y})，预计落入 z={z}；"
            f"{instruction}"
        )
        self._refresh()

    def request_value_estimate(self) -> None:
        """Evaluate the current position without blocking the Tk main thread."""
        if (
            self.terminated
            or not isinstance(self.agent, AlphaZeroAgent)
            or self.observation is None
        ):
            if not isinstance(self.agent, AlphaZeroAgent):
                self.current_value = None
                self.value_var.set("Value network：Alpha-Beta 不适用")
            return

        game_id = self.game_id
        observation = self.observation.copy()
        player = int(self.env.current_player)
        agent = self.agent
        self.value_var.set("Value network：正在计算…")

        def worker() -> None:
            try:
                with self.inference_lock:
                    value = agent.evaluate_state(observation)
                error = None
            except Exception as exc:
                value = None
                error = str(exc)
            self.root.after(
                0,
                self._finish_value_estimate,
                game_id,
                observation,
                player,
                value,
                error,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_value_estimate(
        self,
        game_id: int,
        searched_observation: np.ndarray,
        player: int,
        value: float | None,
        error: str | None,
    ) -> None:
        if game_id != self.game_id or not np.array_equal(searched_observation, self.env.board):
            return
        if error is not None or value is None:
            self.value_var.set(f"Value network：计算失败（{error or '无结果'}）")
            return
        self._record_value_estimate(self.move_count, player, value)

    def _record_value_estimate(self, step: int, player: int, value: float) -> None:
        """Store a raw side-to-move value and plot it from the agent's view."""
        self.current_value = float(value)
        agent_player = -int(self.human_player)
        agent_value = float(value if player == agent_player else -value)
        point = {
            "step": int(step),
            "player": int(player),
            "value": float(value),
            "agent_value": agent_value,
        }
        if self.value_history and int(self.value_history[-1]["step"]) == step:
            self.value_history[-1] = point
        else:
            self.value_history.append(point)
        self.value_var.set(
            f"Value network：{value:+.3f}（当前行动方 {self.PLAYER_NAMES[player]} 视角）"
        )
        self._draw_value_plot()

    def _draw_value_plot(self) -> None:
        self.value_axis.clear()
        self.value_axis.axhline(0.0, color="#95a5a6", linewidth=0.8)
        if self.value_history:
            steps = [int(point["step"]) for point in self.value_history]
            values = [float(point["agent_value"]) for point in self.value_history]
            self.value_axis.plot(steps, values, color="#8e44ad", marker="o", markersize=3)
        self.value_axis.set_xlim(left=0)
        self.value_axis.set_ylim(-1.05, 1.05)
        self.value_axis.set_xlabel("step", fontsize=8)
        self.value_axis.set_ylabel("value", fontsize=8)
        self.value_axis.tick_params(labelsize=7)
        self.value_axis.grid(True, alpha=0.2)
        self.value_figure.tight_layout(pad=1.0)
        self.value_plot_canvas.draw_idle()

    def confirm_agent_move(self) -> None:
        if self.terminated or self.calculating or self.suggested_action is None:
            return
        if self.env.current_player == self.human_player:
            return
        self._apply_action(self.suggested_action)

    def undo_move(self) -> None:
        """Restore exactly one pre-move snapshot from the current game."""
        if not self.position_history:
            return

        snapshot = self.position_history.pop()
        self.game_id += 1  # Invalidates any result still running in the background.
        self.env.board[...] = np.asarray(snapshot["board"], dtype=np.int8)
        self.env.current_player = int(snapshot["current_player"])
        self.env.winner = snapshot["winner"]
        self.env.winning_positions = list(snapshot["winning_positions"])
        self.observation = self.env.board.copy()
        self.terminated = bool(snapshot["terminated"])
        self.move_count = int(snapshot["move_count"])
        self.moves = list(snapshot["moves"])
        stored_value = snapshot["current_value"]
        self.current_value = None if stored_value is None else float(stored_value)
        self.value_history = [dict(point) for point in snapshot["value_history"]]
        self.suggested_action = None
        self.suggested_position = None
        self.calculating = False
        if self.agent is not None:
            self.agent.reset_episode()
        self.suggestion_var.set("Agent 建议：已悔棋，等待重新计算")
        self.value_var.set("Value network：等待重新计算")
        self._draw_value_plot()
        self._refresh()

        if not self.terminated and self.env.current_player != self.human_player:
            self.request_agent_suggestion()
        elif not self.terminated:
            self.request_value_estimate()

    def save_game_record(self) -> None:
        """Serialize the current game to a stable, replay-friendly JSON record."""
        if self.game_started_at is None or self.record_path is None or self.agent is None:
            return

        winner = self.env.winner
        if winner == 0:
            result = "draw"
        elif winner in (1, -1):
            result = f"player_{1 if winner == 1 else 2}_wins"
        else:
            result = "in_progress"

        record = {
            "format": "connect_four_3d_record",
            "version": 1,
            "game_started_at": self.game_started_at.isoformat(timespec="seconds"),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": {
                "board_size": int(self.env.board_size),
                "connect_length": int(self.env.connect_length),
                "human_player": int(self.human_player),
                "agent_player": int(-self.human_player),
                **self.agent_details,
                "action_mapping": "action = board_size * x + y",
                "gravity": "lowest available z; z=0 is bottom",
            },
            "status": {
                "result": result,
                "terminated": bool(self.terminated),
                "winner": None if winner is None else int(winner),
                "current_player": int(self.env.current_player),
                "move_count": int(self.move_count),
            },
            "moves": self.moves,
            "value_history": self.value_history,
            "final_board_xyz": self.env.board.astype(int).tolist(),
            "winning_positions": [
                [int(value) for value in position]
                for position in self.env.winning_positions
            ],
        }

        try:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.record_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法保存棋局记录：\n{exc}")
            return

        messagebox.showinfo("保存成功", f"棋局记录已保存到：\n{self.record_path}")

    def _refresh(self, winner: int | None = None) -> None:
        self._draw_board()
        winner = self.env.winner if winner is None else winner
        if self.terminated:
            result = "平局" if winner == 0 else f"{self.PLAYER_NAMES[winner]} 获胜"
            turn_text = f"对局结束：{result}"
        else:
            role = "真人" if self.env.current_player == self.human_player else str(
                self.agent_details.get("agent", "Agent")
            )
            turn_text = f"当前回合：{self.PLAYER_NAMES[self.env.current_player]}（{role}）"
        self.status_var.set(
            f"{turn_text}　|　双方颜色：Player 1 红 / Player 2 蓝　|　落子数：{self.move_count}"
        )

        self.calculate_button.configure(
            state="normal" if not self.terminated and not self.calculating else "disabled"
        )
        agent_turn = not self.terminated and self.env.current_player != self.human_player
        self.confirm_button.configure(
            state="normal"
            if agent_turn and not self.calculating and self.suggested_action is not None
            else "disabled"
        )
        self.undo_button.configure(
            state="normal" if self.position_history else "disabled"
        )

    def _draw_board(self) -> None:
        winning = {tuple(int(v) for v in position) for position in self.env.winning_positions}
        for z, canvas in enumerate(self.canvases):
            canvas.delete("all")
            canvas.create_text(10, self.MARGIN - 10, text="x", font=("TkDefaultFont", 9, "bold"))
            canvas.create_text(self.MARGIN - 10, 10, text="y", font=("TkDefaultFont", 9, "bold"))
            for index in range(self.env.board_size):
                center = self.MARGIN + index * self.CELL_SIZE + self.CELL_SIZE / 2
                canvas.create_text(center, 12, text=str(index))  # y coordinate
                canvas.create_text(12, center, text=str(index))  # x coordinate
            for x in range(self.env.board_size):
                for y in range(self.env.board_size):
                    x0 = self.MARGIN + y * self.CELL_SIZE
                    y0 = self.MARGIN + x * self.CELL_SIZE
                    x1, y1 = x0 + self.CELL_SIZE, y0 + self.CELL_SIZE
                    canvas.create_rectangle(x0, y0, x1, y1, outline="#657786", width=1)
                    player = int(self.env.board[x, y, z])
                    if player:
                        inset = 6
                        is_winner = (x, y, z) in winning
                        canvas.create_oval(
                            x0 + inset,
                            y0 + inset,
                            x1 - inset,
                            y1 - inset,
                            fill=self.PLAYER_COLORS[player],
                            outline="#f1c40f" if is_winner else "#263238",
                            width=5 if is_winner else 2,
                        )
                    elif self.suggested_position == (x, y, z):
                        # A dashed ring distinguishes advice from an actual piece.
                        canvas.create_oval(
                            x0 + 5,
                            y0 + 5,
                            x1 - 5,
                            y1 - 5,
                            outline="#f39c12",
                            width=4,
                            dash=(5, 3),
                        )
                        canvas.create_text(
                            (x0 + x1) / 2,
                            (y0 + y1) / 2,
                            text="建议",
                            fill="#9a5b00",
                            font=("TkDefaultFont", 8, "bold"),
                        )


def main() -> None:
    root = tk.Tk()
    ConnectFourUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
