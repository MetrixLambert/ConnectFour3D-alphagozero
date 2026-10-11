# ConnectFour3D AlphaZero 工程记录

更新时间：2026-09-28

## 1. 项目目标与总体流程

这是一个面向 5×5×5 立体四子棋的最小 AlphaZero 风格训练工程。它没有“候选模型对冠军模型”的竞技晋级环节，也没有并行自博弈；当前主循环是：

1. 用当前网络执行 MCTS 自博弈；
2. 将 `(规范化状态, MCTS 访问次数策略, 最终胜负)` 放入内存回放池；
3. 从回放池随机采样，以策略交叉熵加 value MSE 更新同一个网络；
4. 每轮覆盖保存 checkpoint。

入口及模块：

| 文件 | 职责 |
| --- | --- |
| `train.py` | 参数解析、设备选择、自博弈/训练循环、checkpoint 保存及恢复 |
| `run_training.py` | 从 `configs/train.json` 构造参数并以非调试方式启动训练 |
| `eval.py` | 载入 checkpoint，轮换先后手，对 RandomAgent 和 AlphaBetaAgent 评估 |
| `run_evaluation.py` | 从 `configs/eval.json` 构造参数并以非调试方式启动评估 |
| `environments/connect_four_3d.py` | 游戏规则、胜负窗口、可逆搜索落子、Gymnasium API、绘图 |
| `models/policy_value_network.py` | 3D 卷积残差策略价值网络 |
| `mcts/mcts.py` | PUCT 搜索、根节点 Dirichlet 噪声、访问次数策略 |
| `self_play/self_play.py` | 单局自博弈和最终 value 标签生成 |
| `self_play/replay_buffer.py` | 定长内存回放池 |
| `training/trainer.py` | 损失、反向传播、checkpoint I/O |
| `utils/symmetry.py` | 水平 x-y 平面的 D4 八种 state/policy 同步数据增强 |
| `evaluation/evaluate.py` | 双方轮换座次的胜/和/负统计 |
| `evaluation/checkpoint_compare.py` | 在相同条件下批量评估 checkpoint 对 Alpha-Beta，并生成汇总表 |
| `agents/` | AlphaZero、Alpha-Beta、随机 agent 及公共基类 |
| `test/test_alpha_zero_core.py` | 网络形状、MCTS 合法策略、单步训练、checkpoint/buffer 往返测试 |

## 2. 游戏状态与动作约定

- 棋盘：NumPy 数组 `(5, 5, 5)`，索引顺序 `[x, y, z]`。
- 编码：空位 `0`，玩家一 `1`，玩家二 `-1`。
- 玩家一永远先手，合法局面中两方棋子数相等时轮到 `1`，玩家一多一子时轮到 `-1`。
- 动作空间：`Discrete(25)`；`action = 5*x + y`，反解为 `x = action // 5`、`y = action % 5`。
- 重力：同一 `(x, y)` 列从最小空闲 `z` 开始填充；顶层 `z=4` 非空即为满列。
- 连线：长度 4；预生成 13 种无反向重复的轴线、平面对角线和空间对角线窗口。
- `env.step` 的获胜 reward 是 `1.0`，否则 `0.0`；AlphaZero 训练不直接使用这个即时 reward，而用整局 winner 生成标签。
- Gymnasium API：`reset() -> (observation, info)`；`step(action) -> (observation, reward, terminated, truncated, info)`。`info` 含 `current_player`、`action_mask`、`winner`。

环境的两组搜索辅助接口：

- `make_move(board, action, player) -> (position, won)` / `unmake_move(board, position)`：MCTS 使用。
- `make_move_with_heights(board, heights, action, player)` / `unmake_move_with_heights(...)`：Alpha-Beta 使用，借助每列高度减少扫描。
- `get_valid_actions_from_board(board)`：从任意棋盘副本取得合法动作。

## 3. 网络与张量形状

`PolicyValueNetwork` 输入规范为 `(batch, 1, 5, 5, 5)`；若直接传 `(batch, 5, 5, 5)`，`forward` 会自动补通道维。

结构为：3D Conv stem → 若干 3D residual blocks → 两个 head：

- policy head 输出 logits `(batch, 25)`，网络内部不做 softmax；MCTS 推理时 softmax，训练时使用 `log_softmax`。
- value head 输出 `(batch,)`，末层 `tanh`，范围 `[-1, 1]`。

stage1 默认结构是 `channels=32`、`num_residual_blocks=2`、value head channels 1、value hidden size 32。stage2 保持 32 channels，将残差块增至 4，并使用 value head channels 8、hidden size 128。stage3 保持 stage2 的32 channels及8/128 value head，只将残差块从4增至8，从而可以复用stem、前4个残差块、policy head和value head的全部权重。checkpoint metadata 保存全部结构参数，`eval.py` 据此重建网络；旧 checkpoint 缺少新字段时回退到 stage1 默认值。

每个新建 ResidualBlock 的第二个 BatchNorm scale 初始化为 0，使新增 block 初始近似恒等映射。stage2 从 stage1 选择性加载名称及形状匹配的 stem、已有 residual blocks 和 policy head，明确排除旧 value head；新 block 和新 value head 使用新初始化。对于value head形状未变的stage3深度扩容，`--reuse-value-head` 会同时迁移所有形状匹配的value head参数；新增的残差块仍为近恒等映射，因此扩容初始的policy/value输出与源网络一致。旧 optimizer 不迁移，replay buffer 则完整继承。

## 4. 玩家视角与 MCTS 符号（关键不变量）

网络输入使用当前行动方视角：

```text
canonical = absolute_board * player_to_move
```

因此网络看到的己方棋子始终为 `+1`，对方为 `-1`。网络 value 表示当前行动方最终获胜的期望。

`MCTSNode.to_play` 是该节点即将行动的玩家；`node.value` 同样采用该节点行动方视角。选择父节点的子节点时使用 `-child.value + UCB exploration`，回传每经过一层便对 value 取反。若某一步已经使落子方获胜，到达的 child 是败方行动视角，所以叶值设置为 `-1.0`。

搜索过程：先展开 root；每次 simulation 沿 PUCT 选择、在 `board.copy()` 上落子，终局或首次未展开节点处求值，再符号交替回传。训练模式只在 root prior 中混入 Dirichlet 噪声。

访问次数转换策略：训练标签始终是 temperature=1 的 MCTS 访问次数分布；自博弈落子
采样单独使用温度调度，默认前 10 手 temperature=1，之后 temperature=0.1。这样后半盘
减少随机败着，但不会把 policy 监督标签压成 one-hot。评估选择访问次数最大动作，平局
随机打破。

自博弈可通过 `--late-game-simulations` 和 `--late-game-simulation-move` 使用分段搜索预算：在指定 ply 前使用 `--simulations`，从指定 ply 起改用更高的后盘预算。该调度只在 `training=True` 的自博弈中生效；评估和 UI 推理始终使用其显式指定的固定 simulations。

## 5. 自博弈、回放与训练

每个行动前记录：

- `state = observation * current_player`；
- `policy = MCTS root children 的访问次数分布`；
- `player = 当时行动方`。

终局后 value 标签为：和棋 `0.0`，否则 `winner * player`。所以每条样本都是其规范化状态中“己方”的最终结果。

启用 `--symmetry-augmentation` 后，每次从 replay buffer 采样的每条样本独立随机选择水平 `x-y` 平面的 8 种 D4 对称变换之一（4 次旋转及其镜像），并同步变换棋盘和 25 维 policy。重力轴 `z` 与 value 标签不变。增强发生在采样后，不扩充或改写 buffer；随机变换使用已随 buffer checkpoint 持久化的 NumPy RNG。

`ReplayBuffer` 是 `deque(maxlen=capacity)`，默认 20,000 条状态样本，不是 20,000 局；采样为无放回随机采样。其 `state_dict` 保存全部样本、容量以及 NumPy RNG 状态。恢复时以 checkpoint 中的容量为准，因此即使新 CLI 传入不同的 `--buffer-size`，已保存 buffer 的容量仍会被保留。
显式传入 `--resize-buffer-on-resume` 时例外：checkpoint 内容和 RNG 恢复后，buffer 会调整为本次 `--buffer-size`，扩容时保留全部现有样本，缩容时由 deque 保留最新样本。未传该开关时继续保持原有兼容语义。

训练可通过 `--recent-sample-window N` 与 `--recent-sample-fraction F` 启用分层采样。
每个 batch 固定保留约 `F` 比例给最新 N 条样本，其余来自更早的历史样本；任一层不足时
从尚未选中的全池样本补足，全程无放回。`F=0`（默认）保持旧版全局均匀采样行为。
采样配置属于训练策略并写入 checkpoint metadata；buffer 本身仍只持久化样本、容量和
RNG，因此旧 checkpoint 可直接恢复。当前 sim200 分支配置使用最近 10,000 条与历史样本
各 50%。

可选 `--ply-sample-fractions F0 F1 F2 F3` 会在上述新旧分层之外，再按棋盘非零棋子数
划分 0–9、10–19、20–29、30+ 四个阶段，并为每个 batch 分配固定比例。各阶段内部继续
遵循 recent/history 比例；某个小层不足时先从同阶段另一新旧层补足，仍不足才从全池尚未
选中的样本补足，全程无放回。四个比例必须非负且总和为 1；未配置时完全保持旧采样行为。
ply 是从 state 直接派生并在内存中缓存，不改变 checkpoint buffer 格式或旧文件兼容性。

训练损失：

```text
loss = -mean(sum(target_policy * log_softmax(policy_logits)))
       + MSE(predicted_value, target_value)
```

优化器为 AdamW（默认 lr `1e-3`、weight decay `1e-4`），梯度范数裁剪到 5。buffer 样本不足一个 batch 时，训练返回全零 loss。

checkpoint 含 `model_state_dict`、`optimizer_state_dict`、`replay_buffer_state_dict`、`metadata`。恢复会载入模型、优化器、replay buffer 内容/容量/采样 RNG，并从 metadata 延续历史 iteration 和累计局数。旧 checkpoint 没有 `replay_buffer_state_dict` 时仍可加载，但首次恢复的 buffer 为空；后续保存会自动写成新格式。

恢复 optimizer 后，`train.py` 会用本次 CLI/JSON 的 `learning_rate` 覆盖各 optimizer parameter group 中随 checkpoint 恢复的旧 LR；AdamW 的动量等其他状态保持不变。实际生效 LR 会打印在启动日志并写入 metadata。

## 6. Agent 与评估

- `AlphaZeroAgent.select_action(observation, training)`：从棋子数量推断当前玩家，运行全新 MCTS；暴露最近的 `get_policy()` 和 root `get_state_value()`。
- `RandomAgent`：在顶层为空的列中均匀随机选择。
- `AlphaBetaAgent`：原地落子/撤销并做 alpha-beta；当前 `_evaluate` 恒为 0，因此深度截止的非终局全部视为中性，它主要能识别搜索深度内的强制胜负，不能作为强启发式基线。
- `evaluate_agents`：偶数编号局 candidate 执先，奇数编号局执后；若 games 为奇数，多出的一局由 candidate 执先。

## 7. CLI 默认值

训练：`train.py` 默认 5 iterations、每轮 2 局、每步 25 次 MCTS simulation、每轮 25 次训练更新、batch 32，保存到 `saved_models/latest.pt`。使用 `--resume` 才加载已有 checkpoint。训练使用 tqdm 显示总 iteration、自博弈局数和优化 step；`--log-interval N` 每隔 N 个本次批次的 iteration 用 `tqdm.write` 输出指标，最后一轮始终输出。

每轮结束都会覆盖 `--checkpoint`，使其始终是最新续训状态。`--snapshot-interval N` 大于 0 时，每逢全局 iteration 被 N 整除，还会在 `--snapshot-dir` 保存完整历史快照。例如主 checkpoint 为 `stage1.pt`、iteration 为 20 时，生成 `stage1_iter_000020.pt`。快照包含模型、优化器、metadata 和完整 replay buffer，可以直接作为 checkpoint 恢复；代价是 buffer 较大时每份快照都会占用较多磁盘。设置 interval 为 0 可禁用。
checkpoint 会先完整写入同目录临时文件，再通过 `os.replace` 原子替换目标路径；并发读取
只能看到旧版或新版完整文件，不会再观察到半写状态。

评估支持两种模式：`eval.py CHECKPOINT` 对单模型分别评估 Random 与 Alpha-Beta；`eval.py --checkpoints A.pt B.pt ...` 在相同 games、MCTS simulations 和 seed 下批量对比历史模型，并输出 iteration、累计训练局数、W/D/L 和胜率表。批量模式通过 `--opponents alpha_beta checkpoint` 选择一种或两种对手；选择 `checkpoint` 时必须同时传入 `--opponent-checkpoint CHAMPION.pt`。未传 `--opponents` 的旧配置仍默认评估 Alpha-Beta，并在存在固定 checkpoint 时追加 checkpoint arena。神经网络 arena 默认用 `--arena-opening-moves 2` 生成成对随机开局：同一开局交换双方座位各下一局，避免 temperature 0 的确定性策略把同一盘棋重复许多次。批量评估使用嵌套 tqdm：外层显示 checkpoint，内层逐局显示进度及当前 W/D/L。`configs/eval.json` 中使用 `checkpoint` 或 `checkpoints` 二选一控制模式，并可用 `opponents` 列表控制对手。

设备均支持 `auto/cpu/cuda`；训练入口会在明确请求不可用 CUDA 时抛错。

## 7.1 桌面 UI

根目录 `run_ui.py` 启动 `ui/connect_four_ui.py`。UI 直接复用本工程的
`ConnectFour3D`、Alpha-Beta、AlphaZero、MCTS 和 `PolicyValueNetwork`，避免维护重复
实现。对局前可选择 stage3 iteration 100 / stage2 iteration 620 预设、自定义 checkpoint 或
Alpha-Beta；选择后弹出对应参数窗口，预设会自动切换到相应部署权重。
AlphaZero 支持 checkpoint、MCTS simulations 和 auto/cuda/cpu，Alpha-Beta 支持
max_depth。默认冠军是纯推理 checkpoint
`saved_models/deployment/stage3_depth8_champion_iter_000130.pt`，只含 model state 与
metadata。
主界面通过“模型参数…”按钮打开统一配置窗口。AlphaZero 可配置 checkpoint、计算设备，
并从 50/100/200/400 中下拉选择 MCTS simulations；Alpha-Beta 可配置 max_depth。
修改只在点击“新对局”重建 agent 时生效，不改变正在进行的棋局。Agent 计算在后台线程
进行，人机棋局写入 `records/games/`。
UI 会在每次局面变化后异步调用 AlphaZero value network，对当前行动方规范化视角直接
估值，并显示 `[-1, 1]` 数值。五层棋盘右下角绘制本局 value-step 曲线；为使曲线正负
含义固定，图中数值统一转换为 Agent 视角。新对局清空曲线，Alpha-Beta 模式不显示估值。
下方“悔棋一步”按钮使用每次正式落子前保存的内存快照，每点击一次恢复一手；棋盘、行动
方、终局/胜线、移动记录和 value 曲线会一起回退。悔棋会使正在运行的旧后台推理结果失效，
并按恢复后的行动方重新计算 value 或 Agent 建议。
“AI 推荐当前方”在真人和 Agent 回合都可使用。真人回合只计算并高亮推荐列，必须由真人
点击棋盘完成落子；“确认 Agent 落子”仍只在 Agent 回合启用，避免推荐功能代替真人操作。

## 8. 当前工程状态与已知限制

- 没有 README、依赖锁定文件或打包配置，跨设备环境尚不可复现。
- `.vscode/launch.json` 提供 Cursor/VS Code 的 stage1 续训、短程 smoke、stage1 评估和 pytest 配置，并显式使用本机 `pyRL` 解释器；该绝对解释器路径在其他设备上需要调整。
- `.vscode/tasks.json` 提供默认的非调试训练任务；在 Cursor 中按 `Ctrl+Shift+B` 即会用本机 `pyRL` 解释器运行 `run_training.py`。日常参数集中维护在 `configs/train.json`，无需重复输入 CLI 参数。
- `Train stage2 from configs/train_stage2.json` Task 使用独立入口。首次运行在 `stage2.pt` 不存在时执行选择性迁移和离线预训练；后续运行自动删除初始化参数并添加 resume。
- 同一 tasks 文件还提供 `Evaluate from configs/eval.json`，评估参数集中维护在 `configs/eval.json`。
- 当前工作区没有可用 `.git` 仓库元数据。
- 可用环境为 Conda `pyRL`，实际解释器是 `C:\Users\Administrator\.conda\envs\pyRL\python.exe`。当前非交互式 PowerShell 没有初始化 `conda`，Windows `py` launcher 也无法发现该环境，需直接调用上述解释器。
- `pyRL` 当前没有安装 pytest。由于现有测试文件只使用普通 `assert`，2026-09-28 通过 `runpy` 直接调用了三个测试函数，全部通过；这等价验证了函数内容，但不是一次标准 pytest 收集运行。
- 三个现有测试覆盖网络输出形状、MCTS 策略合法/归一化、训练一步产生非零损失；尚未系统覆盖 13 类胜线、落子/撤销、终局 value 符号、checkpoint 往返和 CLI。
- 自博弈没有树复用，也没有 batched inference。训练采样已支持水平 D4 对称增强。
- `environments/connect_four_3d.py` 中保留了一段已注释的旧 `get_all_windows` 实现以及不可达的 `return windows`，不影响现有执行但可在后续清理。
- `agents/random_agent.py` 存在重复 import，不影响行为。

## 9. 建议的后续验证顺序

环境具备 pytest 后，建议按以下顺序运行：

```powershell
py -m pytest -q
py train.py --iterations 1 --games-per-iteration 1 --simulations 2 --training-steps 1 --batch-size 2 --channels 8 --residual-blocks 1 --device cpu --checkpoint saved_models/smoke.pt
py eval.py saved_models/smoke.pt --games 2 --simulations 2 --alpha-beta-depth 1 --device cpu
```

训练烟雾测试会写入 checkpoint；如仅需只读验证，先只运行 pytest。

## 10. 2026-09-28 本机验证记录

环境版本：Python 3.11.16、NumPy 2.4.6、PyTorch 2.6.0+cu126、Gymnasium 1.3.0、Matplotlib 3.11.1；`torch.cuda.is_available()` 为 `True`，CUDA runtime 版本 12.6。

直接调用并通过：

- `test_policy_value_network_shapes`
- `test_mcts_returns_a_legal_normalized_policy`
- `test_training_step_updates_from_buffer`

随后以 CPU 执行最小端到端烟雾测试：1 iteration、1 局自博弈、每步 2 simulations、1 training step、batch size 2、channels 8、1 residual block。结果为：

```text
parameters=11,042
games=1, replay samples=14, winner=-1
loss=4.3088, policy_loss=3.1352, value_loss=1.1736
```

生成的临时 checkpoint 能被 `eval.py` 正常载入，并完成对 RandomAgent 与 depth-1 AlphaBetaAgent 各 2 局评估。两组结果碰巧都是 2 胜 0 和 0 负；局数和训练量极小，此结果仅说明端到端流程可运行，不代表模型棋力。

临时 checkpoint 位于系统临时目录 `connectfour3d_codex_smoke.pt`，不属于项目文件。
