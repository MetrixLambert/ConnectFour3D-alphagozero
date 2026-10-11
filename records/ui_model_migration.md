# 冠军模型迁移到 ConnectFour3D UI

目标 UI 工程位于 `tmp/ConnectFour3D-ui`。当前冠军为
`saved_models/checkpoints/stage2_iter_000260.pt`。

## 兼容性结论

两边都使用 5×5×5 棋盘、`board[x, y, z]`、棋子 `1/-1`、空位 `0`、
重力从 `z=0` 向上，以及 `action = 5*x+y`。UI 的环境同时提供 MCTS 所需的
`get_valid_actions_from_board()` 和 `make_move()`，所以棋盘与动作不需要转换。

冠军 checkpoint metadata：channels 32、residual blocks 4、value head channels 8、
value hidden size 128、iteration 260。推理只需要 `model_state_dict` 和 metadata；
完整 checkpoint 中的 optimizer 与 replay buffer 可以保留但不会被读取。

## 需要迁移的文件

从训练工程复制到 UI 工程：

- `models/policy_value_network.py` 与 `models/__init__.py`
- `mcts/mcts.py` 与 `mcts/__init__.py`
- `agents/alpha_zero_agent.py`
- `saved_models/checkpoints/stage2_iter_000260.pt`，建议在 UI 中保存为
  `saved_models/stage2_champion_iter_000260.pt`

UI 已有兼容的 `agents/base_agent.py` 和环境实现，无需替换。

## UI 接入点

在 `ui/connect_four_ui.py` 中：

1. 将 `AlphaBetaAgent` import 替换为 `AlphaZeroAgent`，并 import PyTorch 与
   `PolicyValueNetwork`。
2. 从 checkpoint metadata 构造网络，加载 `model_state_dict`。
3. 用该网络构造 `AlphaZeroAgent`，`num_simulations` 建议先保持训练评估使用的 50。
4. 将界面的 `AlphaBeta max_depth` 改为 `MCTS simulations`。
5. 棋局记录中的 agent 类型、checkpoint、iteration、simulations 和 device 同步更新。
6. 模型应尽量只加载一次；新对局只需 `reset_episode()`，不必重复读取 checkpoint。

核心加载逻辑应等价于：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
metadata = checkpoint.get("metadata", {})
network = PolicyValueNetwork(
    board_size=env.board_size,
    num_actions=env.action_space.n,
    channels=int(metadata.get("channels", 32)),
    num_residual_blocks=int(metadata.get("residual_blocks", 2)),
    value_head_channels=int(metadata.get("value_head_channels", 1)),
    value_hidden_size=int(metadata.get("value_hidden_size", metadata.get("channels", 32))),
)
network.load_state_dict(checkpoint["model_state_dict"])
agent = AlphaZeroAgent(
    env.observation_space,
    env.action_space,
    env=env,
    network=network,
    num_simulations=50,
    device=device,
)
```

UI 已在后台线程调用 `agent.select_action(observation, training=False)`，可继续沿用。
`training=False` 会关闭根节点探索噪声并使用 temperature 0 的确定性落子。

## 运行环境

应使用训练工程的 Conda 环境运行 UI，确保 PyTorch、NumPy、Gymnasium 和 Tkinter
版本兼容：

```powershell
C:\Users\Administrator\.conda\envs\pyRL\python.exe tmp\ConnectFour3D-ui\main.py
```

首次接入后应至少验证：模型能加载、执黑/执白均能落合法动作、落子坐标与 UI 高亮
一致、终局不会继续请求 AI，以及新开局不会沿用上一局的建议状态。

## 2026-09-29 实施结果

迁移已实施到 `tmp/ConnectFour3D-ui`。UI 现在可在 AlphaZero 冠军与 Alpha-Beta
之间选择；选择模型会打开对应参数窗口，也保留独立的“参数设置”按钮。AlphaZero
参数包括 simulations、auto/cuda/cpu 设备和 checkpoint 路径，Alpha-Beta 参数为
max_depth。棋局记录会保存所选 agent 的对应参数。

冠军已导出为 UI 专用推理 checkpoint：
`tmp/ConnectFour3D-ui/saved_models/stage2_champion_iter_000260.pt`。该文件仅含
`model_state_dict` 和 metadata，约 1.46MB，不再包含 optimizer、replay buffer 或对
训练工程 `self_play` 模块的 pickle 依赖。加载使用 `weights_only=True`。

已在 UI 工程目录中独立完成语法检查，并用 CPU/2 simulations 做推理 smoke：
AlphaZero 与 Alpha-Beta 均从空棋盘返回合法动作，checkpoint metadata iteration 为
260。

## 主工程融合

UI 已进一步融合到仓库根目录：

- `run_ui.py` 是根目录运行入口。
- `ui/connect_four_ui.py` 是唯一界面实现，直接导入主工程的 `environments`、
  `agents`、`models` 和 `mcts`，不再复制算法代码。
- 部署权重位于 `saved_models/deployment/stage2_champion_iter_000260.pt`。
- 人机棋局保存到 `records/games/`，与工程设计/训练记录分开。
- Cursor Tasks 新增 `Launch ConnectFour3D UI`。

原始 `tmp/ConnectFour3D-ui` 暂时保留，未删除。根目录版本已完成入口导入、task JSON、
checkpoint weights-only 加载和 canonical 环境合法动作 smoke；主工程 16 项 pytest 通过。

高搜索训练分支 iteration 280 以 62% 战胜原 iteration 260 冠军后，UI 默认模型已更新为
`saved_models/deployment/stage2_sim100_champion_iter_000280.pt`。旧 iteration 260 部署
文件仍保留，可在 AlphaZero 参数窗口中手动选择用于对照。

后续冠军链评估中，iteration 340 战胜 iteration 310，而 iteration 380 未能战胜
iteration 340。UI 默认模型现已更新为
`saved_models/deployment/stage2_sim100_champion_iter_000340.pt`；旧部署文件继续保留。

UI 的对手模型下拉框已改为并列预设，可直接选择高搜索冠军 iteration 340、高搜索
冠军 iteration 280、原冠军 iteration 260、自定义 checkpoint 或 Alpha-Beta。选择前三个
预设时会自动填入相应的 `saved_models/deployment/` 推理权重，然后打开参数窗口；因此旧
冠军仍可直接用于人工对照，无需手动浏览文件。

主界面的循环切换 MCTS 搜索预算按钮随后被统一的“模型参数…”入口取代。点击该按钮会
弹出当前模型的完整配置窗口；AlphaZero 的 simulations 使用 50、100、200、400 只读
下拉菜单，并可同时配置 checkpoint 与计算设备，Alpha-Beta 则配置 max_depth。参数仍在
点击“新对局”后生效，不改变正在进行的棋局。

分层采样分支 iteration 440 在双方 200 simulations 下，以两个独立 seed 分别取得
56% 和 58%，正式取代 iteration 340。UI 默认权重更新为
`saved_models/deployment/stage2_sim200_champion_iter_000440.pt`；预设列表只保留新冠军
440、原冠军 340、自定义 checkpoint 和 Alpha-Beta。iteration 260/280 已从 UI 列表
移除，但其部署文件保留为历史归档，未从磁盘删除。

后续冠军链依次推进至 iteration 580 和 iteration 620。当前 UI 默认预设使用低 LR
冠军 `saved_models/deployment/stage2_sim200_temp08_lr1e4_champion_iter_000620.pt`，第二个
AlphaZero 预设保留上一任冠军
`saved_models/deployment/stage2_sim200_temp08_champion_iter_000580.pt`。原 iteration 340
预设已从 UI 下拉列表移除，但部署文件继续作为历史归档保留，不从磁盘删除。

UI 增加实时 value-network 展示。每次新局面产生后都会在后台对当前行动方规范化棋盘直接
前向估值，状态区显示原始 value 数值；五层棋盘布局右下角的空位嵌入 value-step 曲线。
曲线统一采用 Agent 视角，避免轮到不同玩家时仅因视角约定造成正负交替。新对局清空历史，
Alpha-Beta 模式显示 value network 不适用；保存棋局时同时写入 `value_history`。

下方操作区新增“悔棋一步”。UI 在每次 `env.step` 前保存完整临时快照，按钮每次只恢复
一手，并同步恢复行动方、胜负状态、胜线、棋局 moves 和 value 曲线。恢复动作通过增加
`game_id` 丢弃旧后台计算结果，随后对恢复后的局面重新请求估值或 Agent 建议。

“重新计算建议”调整为“AI 推荐当前方”，并开放到真人回合。推荐仍使用当前所选 Agent
及其配置；真人回合只在棋盘上显示建议位置并给出 action/坐标，确认按钮保持禁用，真人
需要自行点击对应列落子。Agent 回合的原确认流程不变。
