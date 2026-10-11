# Agents 记录

本仓库是 5×5×5、四子连线（Connect Four 3D）的紧凑 AlphaZero 风格实现。

## 开始工作前

- 先阅读 `records/project_architecture.md`，其中记录了棋盘/动作编码、玩家视角、MCTS 价值符号、训练数据流和主要接口。
- 业务入口：`train.py`（自博弈训练）、`eval.py`（对 Random/Alpha-Beta 评估）。
- 桌面 UI 入口：`run_ui.py`；界面实现位于 `ui/`，直接复用主工程的环境、agent、MCTS 和网络。默认部署冠军为 `saved_models/deployment/stage3_depth8_champion_iter_000100.pt`。
- Python 环境：Conda 环境 `pyRL`，解释器位于 `C:\Users\Administrator\.conda\envs\pyRL\python.exe`。当前非交互式 PowerShell 未初始化 `conda`，建议直接调用该解释器。
- 标准测试入口：用 `pyRL` 解释器运行 `python -m pytest -q`；2026-09-29 结果为 17 passed。训练进度见 `records/training_log.md`。
- 非调试训练参数集中在 `configs/train.json`；`run_training.py` 读取它。Cursor 中按 `Ctrl+Shift+B` 会执行 `.vscode/tasks.json` 的默认训练任务，不经过 debugger。
- stage2 配置位于 `configs/train_stage2.json`，入口为 `run_stage2_training.py`；首次从 stage1 iteration 230 选择性迁移，检测到 `saved_models/stage2.pt` 后自动改为 resume。
- 非调试评估参数集中在 `configs/eval.json`；`run_evaluation.py` 读取它，可从 Cursor 的 `Tasks: Run Task` 选择对应任务。
- `configs/eval.json` 使用单个 `checkpoint` 时运行旧的单模型双对手评估；使用 `checkpoints` 列表时批量对比各 checkpoint 与固定 Alpha-Beta 对手的战绩。
- 调试配置位于 `.vscode/launch.json`，包含 stage1 续训、短程 smoke、评估和 pytest。
- Cursor Tasks 包含 `Launch ConnectFour3D UI`，使用 `pyRL` 解释器启动根目录 `run_ui.py`。
- 仓库目前没有 README、`requirements.txt` 或 `pyproject.toml`；已知直接依赖为 NumPy、PyTorch、Gymnasium、Matplotlib、tqdm、pytest（测试）。

## 必须保持的一致性

- 原始棋盘为 `board[x, y, z]`，形状 `(5, 5, 5)`，棋子为 `1/-1`，空位为 `0`；重力沿 `z` 从 0 向上堆叠。
- 动作共 25 个竖直列：`action = 5 * x + y`。
- 网络输入必须是当前行动方视角：`canonical_board = board * player`。
- 网络 value 和每个 MCTS 节点的 value 都表示“该节点行动方”的结果；MCTS 跨层回传必须取反。
- 自博弈标签为当时行动方视角：和棋 `0`，否则 `winner * player_at_state`。
- 对称增强仅允许作用于水平 `x-y` 平面的 D4 八种变换；必须同步变换 state 与 policy，且不能翻转重力轴 `z` 或改变 value。
- 修改上述约定、公共接口或训练流程后，同步更新 `records/project_architecture.md`。

## 修改约束

- 优先保持 Gymnasium 的 `reset` / `step` 返回约定。
- 搜索模拟必须在棋盘副本上进行，不能污染环境或调用者传入的 observation。
- checkpoint 的网络结构依赖 metadata 中的 `channels`、`residual_blocks`。
- stage2 还依赖 metadata 中的 `value_head_channels`、`value_hidden_size`；旧 checkpoint 缺失时分别回退为 1 和 channels。
- 新格式 checkpoint 同时保存 replay buffer（样本、容量、采样 RNG）和全局训练进度；加载旧格式 checkpoint 时须兼容缺失的 buffer 字段。
- 恢复训练时，optimizer 动量等状态来自 checkpoint，但 CLI/JSON 的 `learning_rate` 会显式覆盖 checkpoint 中的旧 LR。
- `checkpoint` 每轮覆盖为最新状态；`snapshot_interval > 0` 时按全局 iteration 在 `snapshot_dir` 保存完整、可恢复的编号快照。
