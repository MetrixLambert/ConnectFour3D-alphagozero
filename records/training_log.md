# 训练记录

## 2026-09-28：stage1

模型文件：`saved_models/stage1.pt`

环境：Python 3.11.16、PyTorch 2.6.0+cu126、NVIDIA GeForce RTX 4060 Ti 8GB。训练前运行 `pytest -q`，结果为 3 passed（2.08 秒）。

### 初始试训

配置：3 iterations、每轮 2 局、15 simulations、每轮 10 training steps、batch 32、channels 32、2 residual blocks、AdamW lr 0.001、CUDA、seed 0。

```text
iteration  games  buffer  W1/D/W2  loss    policy  value
1          2      41      1/0/1    3.2011  2.9654  0.2358
2          4      103     2/0/0    3.4775  2.9546  0.5229
3          6      142     1/0/1    3.1579  2.9026  0.2553
```

初评（10 局/对手、25 simulations）：对 RandomAgent 10-0，对 depth-2 AlphaBetaAgent 0-10。

曾进行一次仅 2 局的恢复测试。因为 replay buffer 不随 checkpoint 保存，恢复后的 buffer 只有 23 条样本，小于 batch 32，loss 为 0，模型没有在该轮更新。这验证了 checkpoint 可加载，同时也确认恢复训练必须先重新积累足够样本。

### 第二阶段

从上述权重恢复。配置：10 iterations、每轮 10 局、50 simulations、每轮 50 training steps、batch 64、channels 32、2 residual blocks、AdamW lr 0.001、CUDA、seed 2。总耗时约 103 秒。

```text
iteration  games  buffer  W1/D/W2  loss    policy  value
1          10     164     4/0/6    2.8322  2.5331  0.2991
2          20     326     4/0/6    2.5707  2.3069  0.2637
3          30     469     7/0/3    2.3987  2.1937  0.2050
4          40     665     6/0/4    2.4274  2.1961  0.2313
5          50     843     4/0/6    2.3755  2.1521  0.2234
6          60     986     7/0/3    2.3302  2.0883  0.2419
7          70     1131    9/0/1    2.2413  2.0247  0.2166
8          80     1251    6/0/4    2.2192  2.0064  0.2128
9          90     1407    6/0/4    2.3059  2.0509  0.2550
10         100    1552    7/0/3    2.2887  2.0433  0.2454
```

复评（20 局/对手、50 simulations、seed 200）：

- RandomAgent：19 胜、0 和、1 负。
- depth-2 AlphaBetaAgent：9 胜、0 和、11 负。

相较初评，对 Alpha-Beta 从 0/10 提升到 9/20，说明训练产生了可观察的战术改进，但样本量仍小，尚不适合把单次结果当作稳定棋力估计。

当前 checkpoint metadata 为：`iteration=10`、`total_games=100`、`channels=32`、`residual_blocks=2`。注意这些计数只覆盖最近一次训练进程，早期试训局数不在 metadata 中；其权重影响已通过 resume 保留。

## 2026-09-28：持久化升级

checkpoint 格式已升级，新增 replay buffer 全部样本、容量和采样 RNG 状态；`train.py --resume` 也会继续累计 iteration 与 total_games。保留对旧 checkpoint 的兼容。

测试结果：`4 passed`。另用两个独立 Python 进程完成 CLI 往返测试：第一批保存 39 条样本；第二批启动时显示 `buffer=39`，结束后保存 68 条，iteration/total_games 均从 1 连续增加到 2。

现有 `saved_models/stage1.pt` 生成于格式升级之前，内部没有历史 replay buffer。它仍能恢复模型、优化器和 metadata，但首次用新版代码恢复时 buffer 将从空开始；该批训练结束并保存后会自动迁移为含 buffer 的新格式。此前已经丢失的 1,552 条样本无法从模型权重反推出。

## 2026-09-28：Cursor 启动与进度显示

新增 `.vscode/launch.json`，可以从 Cursor 的 Run and Debug 面板直接选择 stage1 续训、短程 smoke、评估或 pytest。训练入口新增 tqdm：显示 iteration、自博弈局数和优化 step，并支持 `--log-interval` 控制指标行频率。JSON 校验、4 项 pytest 和两轮 CLI smoke 均通过。

随后新增非调试入口 `run_training.py`、固定配置 `configs/train.json` 和 Cursor 默认任务 `.vscode/tasks.json`。按 `Ctrl+Shift+B` 可直接用 `pyRL` 运行固定配置，不经过 debugger；对应配置转换测试加入后为 5 项 pytest 全部通过。

评估端同步新增 `run_evaluation.py`、`configs/eval.json` 和非调试 Cursor Task。配置解析测试加入后为 6 项 pytest 全部通过。

## 2026-09-28：历史 checkpoint 快照

训练继续每轮覆盖主 checkpoint，同时支持 `snapshot_interval` 和 `snapshot_dir`，按全局 iteration 保存完整编号快照。固定配置默认每 10 轮保存到 `saved_models/checkpoints/stage1_iter_XXXXXX.pt`。快照包含 replay buffer，可独立恢复，但会随 buffer 增长而增大。单元测试及 latest+两个快照的 CLI smoke 验证完成，共 7 项 pytest 通过。

## 2026-09-28：历史 checkpoint 横向评估

新增 `evaluation/checkpoint_compare.py`。`eval.py` 入口现在可接受多个 checkpoint，在相同 seed、MCTS simulations 和 Alpha-Beta depth 下逐个评估，最终打印 W/D/L 与胜率表。`configs/eval.json` 默认选择 iteration 30/60/90/120，每个模型 50 局。两个 checkpoint 的快速 CLI 验证通过，累计 9 项 pytest 通过。

### Iteration 60 与 120 的扩大评估

固定条件：每个 checkpoint 200 局、MCTS 50 simulations、Alpha-Beta depth 3、CUDA、seed 200，轮换先后手。

- iteration 60：113 胜、0 和、87 负，胜率 56.5%。
- iteration 120：107 胜、0 和、93 负，胜率 53.5%。

两者相差 3 个百分点；以 200 局样本量估算，该差异没有统计显著性，不能据此判定 iteration 60 确定优于 iteration 120。两者对该固定对手均约为五五开、略占优。完整评估耗时约 369 秒。

## 2026-09-28：降低学习率继续训练

修复了 resume 时 checkpoint optimizer 覆盖新 learning rate 的问题：恢复 AdamW 状态后，配置 LR 会再次覆盖所有 parameter group，并打印/记录实际值。测试与直接恢复验证均确认 effective LR 为 `0.0003`。

从 iteration 120 继续训练 50 轮至 170；配置保持每轮 10 局、MCTS 50 simulations、50 training steps、batch 64，仅将 LR 从 `0.001` 降到 `0.0003`。总耗时约 446 秒。buffer 在 iteration 146 达到 20,000 容量，最终 checkpoint metadata 为 iteration 170、total_games 1700、learning_rate 0.0003。

前 10 轮平均 loss 约 2.326，后 10 轮约 2.358；policy loss 约从 1.971 降至 1.959，value loss 约从 0.355 升至 0.398。固定评估条件为每个 checkpoint 200 局、MCTS 50、Alpha-Beta depth 3、seed 200：

- iteration 120：107 胜、93 负，胜率 53.5%。
- iteration 170：114 胜、86 负，胜率 57.0%。

iteration 170 高 3.5 个百分点，方向积极且未见退化，但 200 局下仍不足以认定差异具有统计显著性。

### 低学习率阶段的中间快照曲线

统一条件：iteration 120/140/150/160/170，每个 checkpoint 100 局、MCTS 50、Alpha-Beta depth 3、seed 200，轮换先后手。耗时约 446 秒。

| iteration | 胜/和/负 | 胜率 |
| ---: | ---: | ---: |
| 120 | 46/0/54 | 46.0% |
| 140 | 48/0/52 | 48.0% |
| 150 | 46/0/54 | 46.0% |
| 160 | 52/0/48 | 52.0% |
| 170 | 50/0/50 | 50.0% |

未观察到清晰单调上升趋势，各点围绕约 48–50% 波动；100 局下相邻差异不显著。结合此前 iteration 120/170 各 200 局的 53.5%/57.0%，说明评估方差仍较大。当前证据支持“降低 LR 后未明显退化”，但不足以支持“继续相同训练即可稳定提升”。

## 2026-09-29：水平对称数据增强

新增 `utils/symmetry.py`，实现保持重力轴 `z` 不变的水平 D4 八种对称变换。训练时每条采样独立随机变换 state 与 policy，value 不变，且不膨胀 replay buffer。`configs/train.json` 已设置 `symmetry_augmentation: true`，旧 checkpoint 可直接恢复。

逐项验证了 8 种变换的 state/policy 动作对齐、z 坐标保持、连续内存和 policy 概率质量；完整测试为 11 passed。基于 iteration 170 副本的一轮 CPU 恢复烟雾测试成功，metadata 正确写入 `symmetry_augmentation=True`。单 step loss 较高反映旧网络尚未学习对称等价性，不作为正式效果结论。

### 对称增强正式训练：iteration 170–220

配置：50 iterations、每轮 10 局、MCTS 50 simulations、每轮 50 training steps、batch 64、LR 0.0003、buffer 20,000、水平 D4 增强开启。执行环境的首个进程在 iteration 215 后因 20 分钟上限终止，随后从完整 checkpoint 恢复并补跑 5 轮；最终全局进度连续到 iteration 220、total_games 2200。

增强刚启用时 loss 明显上升，随后在相同增强分布内稳定下降：

- iteration 171–180 平均：loss 约 3.638、policy 约 2.743、value 约 0.895。
- iteration 211–220 平均：loss 约 3.323、policy 约 2.564、value 约 0.759。

最终 checkpoint 为 `saved_models/stage1.pt`，完整快照为 `saved_models/checkpoints/stage1_iter_000220.pt`；metadata 记录 iteration 220、total_games 2200、learning_rate 0.0003、symmetry_augmentation true，buffer 为 20,000。后期单轮耗时增加到约 20–30 秒，主要表现为自博弈单局变长；训练优化本身仍较快。

### 对称增强前后固定评估

统一条件：iteration 150/170/200/220，每个 checkpoint 100 局、MCTS 50、Alpha-Beta depth 3、seed 200，轮换先后手。总耗时约 828 秒。

| iteration | 阶段 | 胜/和/负 | 胜率 |
| ---: | --- | ---: | ---: |
| 150 | 增强前 | 46/0/54 | 46.0% |
| 170 | 增强前 | 50/0/50 | 50.0% |
| 200 | 增强后 30 轮 | 79/0/21 | 79.0% |
| 220 | 增强后 50 轮 | 75/0/25 | 75.0% |

增强后的两个 checkpoint 相比 iteration 170 提升约 25–29 个百分点，幅度远大于 100 局的抽样误差，支持对称增强带来了真实棋力提升。iteration 200 与 220 相差 4 个百分点，该差异不显著，不能判定 200 确定优于 220；两者都应保留。

### 对称增强继续训练：iteration 220–250

沿用 LR 0.0003、每轮 10 局、MCTS 50 simulations、50 training steps、batch 64 和水平 D4 增强，再训练 30 轮。总耗时约 818 秒，生成 iteration 230、240、250 完整快照。

- iteration 221–230 平均：loss 3.2751、policy 2.5347、value 0.7403。
- iteration 241–250 平均：loss 3.1929、policy 2.4909、value 0.7020。

三个 loss 分项继续下降。最终 checkpoint metadata 为 iteration 250、total_games 2500、learning_rate 0.0003、symmetry_augmentation true，buffer 保持在容量上限 20,000。

### Iteration 200–250 固定评估

统一条件：iteration 200/220/230/240/250，每个 checkpoint 100 局、MCTS 50、Alpha-Beta depth 3、seed 200，轮换先后手。总耗时约 960 秒。

| iteration | 胜/和/负 | 胜率 |
| ---: | ---: | ---: |
| 200 | 79/0/21 | 79.0% |
| 220 | 75/0/25 | 75.0% |
| 230 | 84/0/16 | 84.0% |
| 240 | 76/0/24 | 76.0% |
| 250 | 79/0/21 | 79.0% |

模型已稳定在对 depth-3 Alpha-Beta 约 75%–80% 的明显优势区间。iteration 230 单点最高，但相邻 checkpoint 没有单调趋势，100 局下不足以证明 230 确定优于其他增强后版本；iteration 220–250 也没有显示继续相同配置即可稳定提高胜率。

## 2026-09-29：stage2 网络迁移

stage2 结构：channels 32、residual blocks 4、value head channels 8、value hidden size 128，共 357,496 参数；stage1 为 122,218 参数。新增 residual blocks 以近似恒等映射初始化。

来源为 `stage1_iter_000230.pt`。迁移加载 38 个兼容张量（stem、原有 blocks、policy head 等），明确跳过 10 个旧 value 张量，完整继承 20,000 条 replay buffer，不继承旧 optimizer。随后以 LR 0.0003、D4 增强在历史 buffer 上完成 500 个 GPU 预训练 step：平均 loss 3.3333、policy 2.5194、value 0.8139。

正式初始化模型保存为 `saved_models/stage2.pt`，metadata 为 stage2 iteration 0，并记录 source iteration 230、source total games 2300、pretraining steps 500 和完整结构字段。两局 CPU 快速评估成功，证明 `eval.py` 能从 metadata 重建新架构。完整测试为 14 passed。

stage2 的主要训练仍采用当前模型 self-play。iteration 230 保留为固定评估/arena 基准，不作为长期训练对手，以避免新模型被旧策略锚定。若后续需要模型晋级机制，应增加候选模型对冠军模型的独立 arena gating。

### stage2 正式训练：iteration 0–50

配置：每轮 10 局 self-play、MCTS 50 simulations、50 training steps、batch 64、LR 0.0003、20,000 replay buffer、水平 D4 增强。训练耗时约 491 秒，最终为 stage2 iteration 50、total games 500，并生成 iteration 10/20/30/40/50 完整快照。

- iteration 1–10 平均：loss 3.2393、policy 2.4957、value 0.7436。
- iteration 41–50 平均：loss 2.9983、policy 2.3628、value 0.6355。
- 最后一轮：loss 2.9472、policy 2.3205、value 0.6267。

训练 loss 三项均呈下降，说明扩容后的新增层和 value head 正在适应 self-play 数据。最终 `saved_models/stage2.pt` 与 `stage2_iter_000050.pt` 均记录 iteration 50、total games 500、LR 0.0003、4 个 residual blocks、value head channels 8、value hidden size 128；replay buffer 保持 20,000 条。

固定评估条件：每个 checkpoint 100 局、MCTS 50、Alpha-Beta depth 3、seed 200，轮换先后手。

| checkpoint | iteration | 胜/和/负 | 胜率 |
| --- | ---: | ---: | ---: |
| stage1 | 230 | 84/0/16 | 84.0% |
| stage2 | 10 | 76/0/24 | 76.0% |
| stage2 | 30 | 83/0/17 | 83.0% |
| stage2 | 50 | 85/0/15 | 85.0% |

stage2 在架构迁移初期先下降，随后恢复并在 iteration 50 略高于来源模型，未发现持续退化。stage2-50 与 stage1-230 仅差 1 个百分点，100 局下不具统计显著性，因此只能判断“已恢复到至少相当水平且趋势积极”，暂不能断言扩容已经带来确定的棋力提升。

### stage2 继续训练：iteration 50–100

保持相同配置续训 50 轮：每轮 10 局 self-play、MCTS 50 simulations、50 training steps、batch 64、LR 0.0003、水平 D4 增强。耗时约 452 秒；新增 500 局，最终 stage2 iteration 100、total games 1000，replay buffer 保持 20,000 条，并生成 iteration 60/70/80/90/100 快照。

- iteration 51–60 平均：loss 2.9342、policy 2.3244、value 0.6099。
- iteration 91–100 平均：loss 2.7561、policy 2.1960、value 0.5601。
- iteration 100：loss 2.7266、policy 2.1749、value 0.5517。

本阶段总 loss、policy loss 和 value loss 的分段均值均继续下降。`saved_models/stage2.pt` 与 `stage2_iter_000100.pt` 已核对为 iteration 100、total games 1000、LR 0.0003、buffer 20,000 条。

### stage2 iteration 50/70/100 对比评估

统一条件：每个候选 100 局、MCTS 50 simulations、seed 200。对 Alpha-Beta 使用 depth 3；对 stage1 iteration 230 的神经网络 arena 中，双方均使用 50 simulations，并采用 2 手成对随机开局，同一开局交换座位各下一局。

| stage2 iteration | 对 Alpha-Beta 胜/和/负 | 胜率 | 对 stage1-230 胜/和/负 | 胜率 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 85/0/15 | 85.0% | 69/0/31 | 69.0% |
| 70 | 85/0/15 | 85.0% | 84/0/16 | 84.0% |
| 100 | 85/0/15 | 85.0% | 82/0/18 | 82.0% |

stage2 三个点对固定 Alpha-Beta 基准持平，但都直接战胜了迁移来源 stage1-230，其中 iteration 70/100 优势很大。iteration 70 与 100 相差 2 个百分点，不足以判定 70 确定更强；当前证据表明 iteration 70 后对这两个对手进入平台期。

最初未加入随机开局的神经网络 arena 曾得到 iteration 50 为 0/100、iteration 70/100 为 100/0 的极端结果。检查发现评估 temperature 为 0 且无 root noise，相同座位会重复确定性对局，所以该结果不代表 100 个独立样本，已弃用。评估入口随后增加 `arena_opening_moves`，以上表格中的直接交锋均来自修正后的成对随机开局结果。

基于这次评估，`stage2_iter_000070.pt` 被指定为后续 arena 的当前冠军。它是独立编号快照，不会被最新 `stage2.pt` 覆盖。

### stage2 继续训练：iteration 100–150

保持相同配置再续训 50 轮：每轮 10 局 self-play、MCTS 50 simulations、50 training steps、batch 64、LR 0.0003、水平 D4 增强。耗时约 436 秒；新增 500 局，最终 stage2 iteration 150、total games 1500，replay buffer 保持 20,000 条，并生成 iteration 110/120/130/140/150 快照。

- iteration 101–110 平均：loss 2.6979、policy 2.1501、value 0.5477。
- iteration 141–150 平均：loss 2.4837、policy 2.0171、value 0.4666。
- iteration 150：loss 2.4596、policy 2.0031、value 0.4565。

本阶段三个 loss 分项的分段均值继续下降，尚未出现优化停滞。`saved_models/stage2.pt` 与 `stage2_iter_000150.pt` 已核对为 iteration 150、total games 1500、LR 0.0003、buffer 20,000 条；iteration 70 冠军快照仍完整。`configs/eval.json` 已预设用 iteration 100/120/150 挑战 iteration 70 冠军，同时继续对 Alpha-Beta depth 3 评估。

### stage2 iteration 100/120/150 冠军挑战

统一条件：每组 100 局、MCTS 50 simulations、seed 200。Alpha-Beta depth 3；神经网络 arena 使用 2 手成对随机开局，同一开局交换双方座位各下一局。

| stage2 iteration | 对 Alpha-Beta 胜/和/负 | 胜率 | 对冠军 iteration 70 胜/和/负 | 胜率 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 85/0/15 | 85.0% | 60/0/40 | 60.0% |
| 120 | 91/0/9 | 91.0% | 62/0/38 | 62.0% |
| 150 | 83/0/17 | 83.0% | 63/0/37 | 63.0% |

三个候选均超过对旧冠军的 55% 晋级线，说明 iteration 70 已可被替换。由于共同对手结果不能决定 iteration 120 与 150 的相对强弱，又补做 iteration 150 挑战 iteration 120：结果为 53/0/47，未达到 55% 晋级线。结合 iteration 120 对 Alpha-Beta 的 91% 明显高于 iteration 150 的 83%，将 `stage2_iter_000120.pt` 指定为新冠军。`configs/eval.json` 的固定神经网络对手已更新为 iteration 120。

### stage2 继续训练：iteration 152–202

启动时 `stage2.pt` 的实际进度已是 iteration 152、total games 1520；本次严格续训配置中的 50 轮，因此终点为 iteration 202，而不是此前根据记录预期的 200。保持每轮 10 局 self-play、MCTS 50 simulations、50 training steps、batch 64、LR 0.0003 和水平 D4 增强。耗时约 460 秒，最终 total games 2020，buffer 保持 20,000 条。

- iteration 153–162 平均：loss 2.4091、policy 1.9587、value 0.4504。
- iteration 193–202 平均：loss 2.2452、policy 1.8729、value 0.3722。
- iteration 202：loss 2.2543、policy 1.8682、value 0.3861。

三个 loss 分项继续下降。`saved_models/stage2.pt` 已核对为 iteration 202、total games 2020、LR 0.0003、buffer 20,000 条；最近的完整编号快照 `stage2_iter_000200.pt` 为 iteration 200、total games 2000。iteration 120 冠军保持不变。`configs/eval.json` 已预设 iteration 150/170/200 挑战 iteration 120，并继续对 Alpha-Beta depth 3 评估。

### stage2 iteration 150/170/200 冠军挑战

统一条件：每组 100 局、MCTS 50 simulations、seed 200。Alpha-Beta depth 3；神经网络 arena 使用 2 手成对随机开局，同一开局交换双方座位各下一局。

| stage2 iteration | 对 Alpha-Beta 胜/和/负 | 胜率 | 对冠军 iteration 120 胜/和/负 | 胜率 |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 83/0/17 | 83.0% | 53/0/47 | 53.0% |
| 170 | 83/0/17 | 83.0% | 56/0/44 | 56.0% |
| 200 | 92/0/8 | 92.0% | 65/0/35 | 65.0% |

iteration 170 刚超过 55% 晋级线；iteration 200 则同时在两个对手上取得本组最佳结果，对旧冠军有 65% 的明显优势，对 Alpha-Beta 也刷新至 92%。因此无需候选间加赛，`stage2_iter_000200.pt` 晋升为新冠军。`configs/eval.json` 的固定对手已更新为 iteration 200，默认候选暂设为持续更新的 `saved_models/stage2.pt`。

### stage2 iteration 230/260/300 冠军挑战

统一条件：每组 100 局、MCTS 50 simulations、seed 200。Alpha-Beta depth 3；神经网络 arena 使用 2 手成对随机开局，同一开局交换双方座位各下一局。本次评估是在 `opponents` 配置开关加入前启动，因此仍同时运行了 Alpha-Beta 和 checkpoint 两类对手。

| stage2 iteration | 对 Alpha-Beta 胜/和/负 | 胜率 | 对冠军 iteration 200 胜/和/负 | 胜率 |
| ---: | ---: | ---: | ---: | ---: |
| 230 | 93/0/7 | 93.0% | 64/0/36 | 64.0% |
| 260 | 94/0/6 | 94.0% | 61/0/39 | 61.0% |
| 300 | 93/0/7 | 93.0% | 58/0/42 | 58.0% |

三个候选均超过对 iteration 200 的 55% 晋级线，说明这一百轮训练产生了实际棋力提升；Alpha-Beta 胜率均已在 93%–94%，区分度有限。共同对手战绩不能可靠决定三个候选的相对强弱。按时间顺序的冠军晋级流程，iteration 230 先成为临时冠军，随后应由 iteration 260 直接挑战 230，胜者再接受 iteration 300 挑战。`configs/eval.json` 已配置为仅运行 iteration 260 对 iteration 230 的 checkpoint arena。

iteration 260 随后以 56/0/44、56% 的成绩直接战胜 iteration 230，刚好超过 55% 晋级线，因此 iteration 260 晋升为临时冠军。该优势较小，不能理解为悬殊差距，但按预先确定的规则应予晋级。`configs/eval.json` 已继续更新为仅运行 iteration 300 挑战 iteration 260 的最终 checkpoint arena。

最终 iteration 300 挑战 iteration 260 的胜率为 48%，未达到 55% 晋级线，且低于 50%。因此 `stage2_iter_000260.pt` 保持为正式冠军，iteration 300 不晋级。这也说明 iteration 260–300 期间训练 loss 的继续下降没有转化为对当前冠军的实际棋力提升。`configs/eval.json` 的固定对手保持 iteration 260，默认候选恢复为持续更新的 `saved_models/stage2.pt`，供下一批训练后直接挑战冠军。

### 100 simulations / 50,000 buffer 分支：iteration 230–280

从原主线 `stage2_iter_000230.pt` 复制完整模型、optimizer 和 20,000 条 replay 样本，建立独立 checkpoint `stage2_sim100_from230.pt`。配置只改变 self-play MCTS simulations（50→100）和 replay buffer 容量（20,000→50,000），其余保持每轮 10 局、50 training steps、batch 64、LR 0.0003、D4 增强。新增显式 `resize_buffer_on_resume`，因此默认恢复行为仍保留 checkpoint 容量，只有该实验分支主动扩容。

训练 50 轮至全局 iteration 280，耗时约 833 秒，新增 500 局。buffer 从 20,000 条自然增长到 25,482 条、容量 50,000；没有复制旧样本填充。生成 `stage2_sim100_from230_iter_000240/250/260/270/280.pt` 五个完整快照。

- iteration 231–240 平均：loss 2.1547、policy 1.8281、value 0.3267。
- iteration 271–280 平均：loss 2.1558、policy 1.8347、value 0.3211。
- iteration 280：loss 2.1487、policy 1.8353、value 0.3133。

本阶段 loss 总体横盘，不能单凭 loss 判断高搜索配置是否有效。`configs/eval.json` 已配置为让分支 iteration 240/260/280 在相同 50 simulations 推理预算下挑战当前冠军 `stage2_iter_000260.pt`，以隔离训练质量提升，而不是给候选更多评估算力。

固定 arena 结果（每个候选 100 局、双方均为 50 simulations、2 手成对随机开局、seed 200）：

| 高搜索分支 iteration | 对原主线冠军 iteration 260 胜/和/负 | 胜率 |
| ---: | ---: | ---: |
| 240 | 46/0/54 | 46.0% |
| 260 | 42/0/58 | 42.0% |
| 280 | 62/0/38 | 62.0% |

分支早期先退化，至 iteration 280 才出现明显跃升；iteration 280 超过 55% 晋级线，成为新冠军。这个结果说明 100 simulations 与扩容 buffer 的组合最终产生了实际棋力提升，但提升并不随训练轮数单调出现，也无法从横盘 loss 中直接读出。新冠军已导出为纯推理文件 `saved_models/deployment/stage2_sim100_champion_iter_000280.pt`，根目录 UI 默认模型同步更新。`configs/eval.json` 的固定对手已改为该冠军。

### 高搜索分支继续训练：iteration 280–380

保持 100 self-play simulations、50,000 buffer、每轮 10 局、50 training steps、batch 64、LR 0.0003 和 D4 增强不变，连续训练 100 轮。耗时约 1,637 秒，新增 1,000 局；最终 iteration 380、total games 3800，buffer 从 25,482 增长至 36,203/50,000。

- iteration 281–290 平均：loss 2.1344、policy 1.8215、value 0.3129。
- iteration 371–380 平均：loss 2.1011、policy 1.8127、value 0.2884。
- iteration 380：loss 2.0842、policy 1.8132、value 0.2710。

分段均值缓慢下降，主要改善来自 value loss；policy loss 变化较小。`stage2_sim100_from230.pt` 与 iteration 380 编号快照已核对为 iteration 380、total games 3800、LR 0.0003、buffer 36,203、容量 50,000。`configs/eval.json` 已预设 iteration 310/340/380 在统一 50 simulations 评估预算下挑战当前冠军 iteration 280。

固定 arena 首轮结果（每个候选 100 局、双方 50 simulations、2 手成对随机开局、seed 200）：iteration 310/340/380 对 iteration 280 分别为 55/0/45、64/0/36、55/0/45，三个候选都达到 55% 晋级线。随后按时间顺序直接晋级：iteration 340 以 56/0/44 战胜临时冠军 iteration 310；iteration 380 再挑战 iteration 340 时仅为 46/0/54，未晋级。

最终新冠军为 `stage2_sim100_from230_iter_000340.pt`。这再次表明棋力随训练轮数并不单调：iteration 380 的 loss 更低，但实际弱于 iteration 340。纯推理冠军已导出到 `saved_models/deployment/stage2_sim100_champion_iter_000340.pt`，根目录 UI 默认模型同步更新。评估配置的固定对手已更新为 iteration 340，默认候选为持续更新的高搜索分支 checkpoint。

### 200 simulations / 温度调度分支：从 iteration 340 开始

基于两盘人工对局暴露出的早期多方向双杀判断问题，建立独立分支
`stage2_sim200_from340.pt`。它从 iteration 340 的完整 checkpoint（包括 optimizer 与
replay buffer）继续，计划先训练 50 iterations；自博弈搜索从 100 增至 200 simulations，
其余保持每轮 10 局、50 training steps、batch 64、50,000 buffer、LR 0.0003 与 D4 增强。

同时将自博弈探索改为温度调度：前 10 手 temperature=1.0，之后为 0.1。MCTS 原始访问
次数分布仍作为 policy 训练标签，温度只影响实际落子采样，避免低温阶段产生 one-hot
监督标签。配置入口为 `configs/train_stage2_sim200_from340.json`，运行入口为
`run_stage2_sim200_training.py`。

该批 50 iterations 已完成至全局 iteration 390，耗时约 1,630 秒（27 分钟），新增
500 局，replay buffer 从 31,966 增至 37,543/50,000。iteration 341–350 的平均
loss/policy/value 为 2.1067/1.8071/0.2995；iteration 381–390 为
2.1254/1.8420/0.2833。value loss 略有改善，但总 loss 与 policy loss 没有下降，不能据此
判定棋力提升，必须通过 arena 评估。自博弈结果高度偏向先手，绝大多数 iteration 为
Player 1 取得 9–10/10 胜，这可能反映游戏本身先手优势，也提示后续评估必须继续采用
交换座次的成对开局。

最终主 checkpoint 与编号快照
`saved_models/checkpoints/stage2_sim200_from340_iter_000390.pt` 均已核对为 iteration 390、
total games 3900、buffer 37,543，温度切换手数 10、后段温度 0.1。完整测试为 18 passed。

固定 arena 首轮评估使用双方统一 50 simulations、100 局、2 手成对随机开局及 seed 200，
候选 iteration 350/370/390 对原冠军 iteration 340 的结果分别为 47/0/53、47/0/53、
50/0/50。没有候选达到 55% 晋级线，因此 iteration 340 继续保持冠军。iteration 390 是
该分支相对最好的候选，但目前只能视为与冠军持平，不能认为 200 simulations 自博弈和
温度调度已经带来棋力提升。

随后将双方评估预算同时提高到 200 simulations，使其与新分支自博弈预算一致。
iteration 390 对 iteration 340 的 100 局结果为 52/0/48（52%），较 50 simulations 下
的 50% 略高，但仍未达到 55% 晋级线，且 100 局下这一差距不足以证明稳定优势。因此
iteration 340 继续保持正式冠军，iteration 390 暂作为高搜索候选保留。

下一阶段首先加入 replay buffer 分层采样。`stage2_sim200_from340` 配置设为每个 batch
50% 来自最新 10,000 条样本，50% 来自更早历史样本；任一层不足时自动从未选样本补足。
默认训练配置仍为 0% 近期配额，保持旧行为。此改动不改变 checkpoint buffer 格式，旧
checkpoint 可直接恢复；新的采样参数会写入 metadata。

### 分层采样续训：iteration 390–440

从 iteration 390 继续 50 轮，保持 200 simulations、温度 1.0→0.1、每轮 10 局、50
training steps、batch 64、LR 0.0003 和 D4 增强，并启用每 batch 50% 最新 10,000 条、
50% 更早历史样本的分层采样。训练耗时约 1,547 秒（25 分 44 秒），新增 500 局，
buffer 从 37,543 增至 42,906/50,000。

- iteration 391–400 平均 loss/policy/value：2.1003/1.8464/0.2540。
- iteration 431–440 平均 loss/policy/value：2.0485/1.8274/0.2212。
- iteration 440：loss 2.0990、policy 1.8411、value 0.2580。

首尾十轮三项均下降，尤其 value loss；相比上一批横盘，分层采样至少改善了优化指标，
但实际棋力仍须由 arena 验证。主 checkpoint 与
`stage2_sim200_from340_iter_000440.pt` 均核对为 iteration 440、total games 4400、buffer
42,906、近期采样 0.5@10,000；完整测试 20 passed。

固定 arena 使用双方 200 simulations、100 局、2 手成对随机开局和 seed 200，让
iteration 400/420/440 挑战 iteration 340 冠军。结果分别为 51/0/49、51/0/49、
56/0/44。只有 iteration 440 达到预设 55% 晋级线，说明分层采样后期出现了实际棋力
提升信号；但 56% 优势仍较小，正式替换 UI 冠军前建议用不同 seed 再进行一组确认赛。

使用独立随机开局 seed 201 再进行 100 局确认赛，双方仍为 200 simulations，iteration
440 再次以 58/0/42（58%）战胜 iteration 340。两组合计为 114/0/86，即 57%；两次均
超过 55% 晋级线，因此 iteration 440 正式晋升为新冠军。`configs/eval.json` 的固定对手
已更新为该快照，默认候选恢复为持续更新的 sim200 主 checkpoint；UI 暂未替换，等待
明确迁移指令。

### 分层采样继续训练：iteration 440–490

保持上一阶段全部参数不变继续 50 轮：200 simulations、温度 1.0→0.1、每轮 10 局、
50 training steps、batch 64、LR 0.0003、D4 增强，以及 50% 最新 10,000 条 + 50%
历史样本的分层采样。耗时约 1,625 秒（27 分 1 秒），新增 500 局，buffer 从 42,906
增至 48,467/50,000。

- iteration 441–450 平均 loss/policy/value：2.0562/1.8330/0.2232。
- iteration 481–490 平均 loss/policy/value：2.0546/1.8289/0.2257。
- iteration 490：loss 2.0709、policy 1.8315、value 0.2394。

本批 policy 略降，但总 loss 和 value loss 基本横盘，优化指标已不如 390–440 阶段明确。
主 checkpoint 与 `stage2_sim200_from340_iter_000490.pt` 均核对为 iteration 490、total
games 4900、buffer 48,467；完整测试 20 passed。训练期间只读检查数次撞上主 checkpoint
覆盖写入窗口而临时无法加载，写完后均恢复正常；后续应将 checkpoint 保存改为临时文件
完成后原子替换，以降低并发读取和意外中断风险。

固定 arena 使用双方 200 simulations、100 局、2 手成对随机开局和 seed 200，让
iteration 460/480/490 挑战 iteration 440 冠军。结果分别为 51/0/49、58/0/42、
52/0/48。只有 iteration 480 达到 55% 晋级线；iteration 490 再次说明 loss 或训练轮数
不与棋力单调对应。iteration 480 暂为候选新冠军，正式晋级前应使用独立 seed 201 再做
一组 100 局确认赛。

独立 seed 201 确认赛中，iteration 480 对 iteration 440 为 54/0/46（54%）。该单组略低
于 55%，但方向仍为正；与 seed 200 的 58/0/42 合计为 112/0/88，即 56%，达到合并
晋级线。因此 iteration 480 正式晋升为新冠军。`configs/eval.json` 的固定对手已更新为
iteration 480，默认候选恢复为 sim200 主 checkpoint；UI 暂时仍使用 iteration 440。

下一阶段从正式冠军 iteration 480 的完整快照建立独立精调分支
`stage2_sim200_lr1e4_from480.pt`，只将 LR 从 0.0003 降至 0.0001，计划训练 30 轮至
iteration 510；200 simulations、温度 1.0→0.1、分层采样及其他参数保持不变。若后续
切换到温度调度实验，LR 应恢复为 0.0003，以免同时混入两个变量。checkpoint 保存同时
改为同目录临时文件写完后原子替换。

低 LR 精调首次运行完成 iteration 481–500 后，在 iteration 501 保存阶段遇到 Windows
对目标 checkpoint 的瞬时文件锁，`os.replace` 返回 WinError 5，训练安全停止；最后完整
恢复点为 iteration 500。原子替换随后增加最多 20 次、每次间隔 0.1 秒的 PermissionError
重试，续跑配置临时调整为剩余 10 轮，以保证总训练量仍为 30 iterations。

修复后从 iteration 500 恢复并完成至 iteration 510，总计仍为从冠军 480 开始的 30 轮
精调；buffer 达到 50,000 容量并开始滚动淘汰旧样本。iteration 481–490 平均
loss/policy/value 为 2.0713/1.8280/0.2433，iteration 501–510 为
2.1228/1.8377/0.2851，后十轮三项均上升，低 LR 并未表现出更好的优化趋势。主 checkpoint
与 `stage2_sim200_lr1e4_from480_iter_000510.pt` 已核对为 iteration 510、total games
5100、buffer 50,000、LR 0.0001；完整测试 20 passed。是否有棋力收益仍需用 iteration
490/500/510 挑战 iteration 480 判断。配置的默认批次数已恢复为 30；若下一阶段改温度，
按约定应恢复 LR 0.0003。

低 LR 分支的固定 arena 首轮评估采用双方 200 simulations、100 局、2 手成对随机开局和 seed 200；iteration 490/500/510 对正式冠军 iteration 480 的结果依次为 55/0/45、57/0/43、59/0/41。三个时间点均达到预设 55% 晋级线，且胜率随训练轮数连续上升，说明尽管末段训练 loss 上升，LR 0.0001 的 30 轮精调仍转化为了实际棋力收益。iteration 510 是本分支最佳候选，但在正式晋升前仍需使用独立 seed 201 进行 100 局确认赛。`configs/eval.json` 已设置为仅运行 iteration 510 对 iteration 480 的 seed 201 确认赛。

按用户决定跳过独立 seed 确认赛，iteration 510 直接晋升为正式冠军。纯推理权重导出为 `saved_models/deployment/stage2_sim200_champion_iter_000510.pt`，UI 默认预设与评估固定对手均同步更新。下一阶段从该完整 checkpoint 建立独立分支 `stage2_sim200_temp08_from510.pt`，将前 10 手自博弈采样温度从 1.0 降至 0.8，后段仍为 0.1；LR 按先前约定恢复至 0.0003，其余参数保持不变，计划训练 50 轮至 iteration 560。

温度 0.8 分支已完成 iteration 511–560，共新增 500 局。首次进程受 1 小时执行上限影响，在 iteration 560 的未完成自博弈中终止；最后完整恢复点为 iteration 559，随后从该 checkpoint 补跑 1 轮，因此最终训练量仍严格为 50 iterations。最终 checkpoint 与 iteration 560 快照均核对为 iteration 560、total games 5600、buffer 50,000、LR 0.0003、温度调度 0.8→0.1（第 10 手切换）。iteration 511–520 平均 loss/policy/value 为 2.2290/1.8741/0.3549，iteration 551–560 为 2.5678/2.0328/0.5350，三项均显著上升；这可能是降低温度后的数据分布变化与恢复较高 LR 共同造成，是否退化仍须 arena 判断。评估配置已选择 iteration 520/540/560 在双方 200 simulations 下挑战冠军 iteration 510。

固定 arena 采用双方 200 simulations、100 局、2 手成对随机开局和 seed 200。iteration 520/540/560 对冠军 iteration 510 分别为 50/0/50、65/0/35、65/0/35。温度分支早期尚未提升，但 iteration 540 与 560 都形成了远超 55% 晋级线的明显优势，证明 loss 上升在这里主要反映训练数据分布和难度变化，不能解读为棋力退化。由于 540 与 560 对共同对手战绩相同，下一场应由 iteration 560 直接挑战 iteration 540，决定本分支冠军；`configs/eval.json` 已按此准备。

iteration 560 随后直接挑战 iteration 540，结果为 54/0/46，未达到 55% 晋级线。iteration 560 略占优势但证据不足，因此按既定规则由 iteration 540 成为正式冠军；它对上一任冠军 iteration 510 的战绩为 65/0/35。纯推理权重已导出为 `saved_models/deployment/stage2_sim200_temp08_champion_iter_000540.pt`，UI 默认预设及评估固定对手同步更新。评估候选恢复为该温度分支的持续更新 checkpoint，供后续训练直接挑战 iteration 540。

下一批沿温度 0.8 分支的最新完整 checkpoint iteration 560 继续训练 30 轮至 iteration 590。虽然 560 未以 55% 晋级，但其对 540 为 54/0/46，没有明确退化，继续使用它可保留额外自博弈数据与 optimizer 状态。训练参数保持 200 simulations、温度 0.8→0.1、LR 0.0003、50,000 buffer、分层采样 0.5@10,000、每轮 10 局与 50 optimization steps 不变；正式冠军在评估前仍为 iteration 540。

该批 30 轮已完成至 iteration 590，耗时约 2,291 秒（38 分 11 秒），新增 300 局；最终主 checkpoint 和 iteration 590 快照均核对为 iteration 590、total games 5900、buffer 50,000、LR 0.0003、温度 0.8→0.1。iteration 561–570 平均 loss/policy/value 为 2.6098/2.0683/0.5414，iteration 581–590 为 2.6183/2.0545/0.5639：总 loss 基本横盘，policy 略降，value 略升。仅凭训练指标无法判断棋力变化；`configs/eval.json` 已设置 iteration 570/580/590 在双方 200 simulations 下挑战正式冠军 iteration 540。

固定 arena 首轮结果为 iteration 570/580/590 对正式冠军 iteration 540 分别取得 52/0/48、58/0/42、59/0/41。iteration 580 与 590 均超过 55% 晋级线，说明 iteration 560–590 的继续训练仍产生实际棋力收益；共同对手成绩不能可靠区分二者，因此评估配置已改为 iteration 590 直接挑战 iteration 580。

iteration 590 直接挑战 iteration 580 的结果为 50/0/50，未达到 55% 晋级线，也没有显示方向性优势。因此选择较早且已以 58/0/42 战胜上一任冠军 540 的 iteration 580 晋升为正式冠军；iteration 590 保留为后续续训起点。纯推理权重已导出为 `saved_models/deployment/stage2_sim200_temp08_champion_iter_000580.pt`，UI 默认预设及评估固定对手同步更新。

下一阶段从 iteration 590 的完整快照建立独立分支 `stage2_sim200_temp08_lr1e4_from590.pt`，只将 LR 从 0.0003 降至 0.0001，保持 200 simulations、温度 0.8→0.1、50,000 buffer、分层采样 0.5@10,000、每轮 10 局和 50 optimization steps 不变，计划训练 30 轮至 iteration 620。正式冠军在 arena 验证前仍为 iteration 580。

低 LR 分支已完成 iteration 591–620，耗时约 2,498 秒（41 分 38 秒），新增 300 局。最终主 checkpoint 与 iteration 620 快照均核对为 iteration 620、total games 6200、buffer 50,000、LR 0.0001、温度 0.8→0.1。iteration 591–600 平均 loss/policy/value 为 2.6411/2.0612/0.5799，iteration 611–620 为 2.6663/2.0468/0.6196：policy 略降，但 value 与总 loss 上升，训练指标未表现出明确改善。棋力仍需 arena 验证；`configs/eval.json` 已设置 iteration 600/610/620 在双方 200 simulations 下挑战正式冠军 iteration 580。

固定 arena 首轮使用双方 200 simulations、100 局、2 手成对随机开局和 seed 200；iteration 600/610/620 对冠军 iteration 580 分别为 52/0/48、50/0/50、56/0/44。只有 iteration 620 达到 55% 晋级线，但优势较小，因此自动追加独立 seed 201 的 100 局确认赛，再按两组累计结果决定是否晋升。

独立 seed 201 确认赛中，iteration 620 再次以 58/0/42 战胜 iteration 580。两组累计为 114/0/86，即 57%；两个 seed 均超过 55% 晋级线，因此 iteration 620 正式晋升为新冠军。纯推理权重已导出为 `saved_models/deployment/stage2_sim200_temp08_lr1e4_champion_iter_000620.pt`，UI 默认预设和评估固定对手同步更新；评估 seed 恢复为 200。

为增强中后期训练，新增按 ply 阶段固定配额的 replay 采样，并从冠军 iteration 620 完整快照建立独立分支 `stage2_sim200_ply_stratified_from620.pt`。计划比例为 0–9 手 35%、10–19 手 30%、20–29 手 25%、30+ 手 10%；每个阶段内部仍按最新 10,000 条与历史样本各 50% 采样。其余参数保持 LR 0.0001、200 simulations、温度 0.8→0.1、50,000 buffer、每轮 10 局和 50 optimization steps，默认配置计划训练 30 轮至 iteration 650。

阶段分层分支的首批训练计划随后调整为 50 轮，从 iteration 620 训练至 iteration 670；采样比例及其他参数保持不变。

该批 50 轮已完成，耗时约 3,716 秒（61 分 56 秒），新增 500 局。主 checkpoint 与 iteration 670 快照均核对为 iteration 670、total games 6700、buffer 50,000/50,000、LR 0.0001，ply 采样比例为 `[0.35, 0.30, 0.25, 0.10]`。iteration 621–630 平均 loss/policy/value 为 2.9589/2.3189/0.6401；iteration 661–670 为 3.0026/2.3625/0.6401。新采样分布更集中于中后盘，因此 loss 不应与旧分支横向直接比较；在本分支内首尾对比时，value loss 持平而 policy loss 上升，未显示出明确的优化改善，实际棋力必须由 arena 判断。`configs/eval.json` 已设置 iteration 630/650/670 在双方 200 simulations 下挑战当前冠军 iteration 620。

阶段分层分支的固定 arena 评估使用双方 200 simulations、100 局、2 手成对随机开局和 seed 200。iteration 630/650/670 对当前冠军 iteration 620 的战绩分别为 50/0/50、50/0/50、47/0/53。没有候选达到 55% 晋级线，而且训练后期 iteration 670 略有回落，因此不进行确认赛，iteration 620 继续保持正式冠军。这说明首批 ply 比例采样未带来可验证的整体棋力收益，后续不应在当前参数下盲目续训。

下一阶段回到冠军 iteration 620 建立独立动态搜索分支 `stage2_dynamic_sim_from620.pt`，关闭 ply 固定配额，恢复仅使用 50% 最近 10,000 条 + 50% 历史样本的 replay 采样。自博弈第 0–19 手使用 200 simulations，第 20 手起使用 400 simulations，以提高中后盘 policy/value 目标质量；评估预算仍固定为 200。其他参数保持 LR 0.0001、温度 0.8→0.1、50,000 buffer、每轮 10 局和 50 optimization steps，计划训练 30 轮至 iteration 650。

动态搜索分支已完成 iteration 621–650，耗时约 3,094 秒（51 分 30 秒），新增 300 局。主 checkpoint 及 iteration 630/640/650 快照均已成功加载核对；最终状态为 iteration 650、total games 6500、buffer 50,000/50,000、LR 0.0001、自博弈搜索 `200→400@ply20`、ply 固定配额关闭。iteration 621–630 平均 loss/policy/value 为 2.6833/2.0428/0.6405；iteration 641–650 为 2.7471/2.0986/0.6485，三项均略升，训练指标没有显示明确改善。`configs/eval.json` 已设置 iteration 630/640/650 使用统一 200 simulations 挑战当前冠军 iteration 620，以验证更高质量后盘搜索是否转化为棋力收益。

动态搜索分支的 seed 200 固定 arena 使用双方 200 simulations、100 局和 2 手成对随机开局；iteration 630/640/650 对冠军 iteration 620 分别为 51/0/49、46/0/54、55/0/45。iteration 650 恰好达到晋级线，因此自动追加 seed 201 的100局确认赛，结果为54/0/46。两组合计为109/0/91，即54.5%；两个 seed 都表现为正向优势，说明动态后盘搜索可能有小幅收益，但合并胜率略低于既定55%晋级线，证据不足以替换正式冠军。iteration 620 继续保持冠军，iteration 650 作为有希望的动态搜索候选保留；评估 seed 已恢复为200。

由于 iteration 650 在两个独立 seed 下都对 iteration 620 呈正向优势，继续沿动态搜索分支训练 30 轮至 iteration 680。参数保持 `200→400@ply20`、LR 0.0001、温度 0.8→0.1、recent/history 50%/50%、50,000 buffer、每轮10局和50 optimization steps不变；计划保存 iteration 660/670/680 快照。正式冠军在新一轮 arena 前仍为 iteration 620。

该批已完成 iteration 651–680，耗时约 2,814 秒（46 分 49 秒），新增300局。主 checkpoint 及 iteration 660/670/680 快照均已成功加载；最终状态为 iteration 680、total games 6800、buffer 50,000/50,000、LR 0.0001、`200→400@ply20`、ply 固定配额关闭。iteration 651–660 平均 loss/policy/value 为 2.7736/2.1194/0.6543；iteration 671–680 为 2.8183/2.1452/0.6731，三项继续上升，表明优化指标没有改善，也提示长时间持续训练可能已接近或越过该分支的最佳点。`configs/eval.json` 已设置 iteration 660/670/680 在统一200 simulations下挑战正式冠军 iteration 620；若有候选达到或接近55%，后续应自动追加独立 seed 确认或与 iteration 650 直接对比。

续训批次的 seed 200 固定 arena 使用双方200 simulations、100局及2手成对随机开局。iteration 660/670/680 对正式冠军 iteration 620 分别为48/0/52、51/0/49、50/0/50。没有候选达到或接近55%晋级线，且都低于 iteration 650 在同一 seed 下的55%，因此不追加确认赛或候选间决胜。本批30轮未产生进一步收益，iteration 650 仍是动态搜索分支最佳候选，iteration 620 继续保持正式冠军。评估配置已回到 iteration 650 对 iteration 620，供必要时复测。

按用户要求清理两次失败实验的模型文件：已删除 `stage2_sim200_ply_stratified_from620*` 和 `stage2_dynamic_sim_from620*` 的主 checkpoint 及全部编号快照，共13个文件、约632 MiB；实验配置、代码与评估记录保留供追溯。正式冠军 iteration 620 的完整 checkpoint 与部署模型均已确认未被删除。

新建数据新鲜度分支 `stage2_recent75_from620.pt`，从正式冠军 iteration 620 的完整训练状态开始。唯一核心变量是将每个batch的最近10,000条样本比例从50%提高到75%，其余25%来自更早历史样本；关闭ply固定配额，恢复固定200 simulations。其他参数保持 LR 0.0001、温度 0.8→0.1、50,000 buffer、每轮10局和50 optimization steps，计划训练30轮至 iteration 650，保存630/640/650快照。

数据新鲜度分支已完成 iteration 621–650，耗时约 2,236 秒（37 分 12 秒），新增300局。主 checkpoint 及 iteration 630/640/650 快照均已加载核对；最终状态为 iteration 650、total games 6500、buffer 50,000/50,000、LR 0.0001、recent sampling 0.75@10,000、ply配额关闭、固定200 simulations。iteration 621–630 平均 loss/policy/value 为 2.7910/2.0706/0.7205；iteration 641–650 为 2.8157/2.1259/0.6898。value loss 有所下降，policy loss 上升导致总 loss 略升，仅凭优化指标不能判断棋力。`configs/eval.json` 已设置 iteration 630/640/650 使用统一200 simulations、100局、seed 200 挑战正式冠军 iteration 620。

数据新鲜度分支的 seed 200 固定 arena 结果为：iteration 630/640/650 对正式冠军 iteration 620 分别取得49/0/51、54/0/46、49/0/51。iteration 640 距55%晋级线仅1局，因此自动追加 seed 201 的100局确认赛，结果为53/0/47。两组合计为107/0/93，即53.5%；两个seed均为小幅正向，说明提高数据新鲜度可能有微弱收益，但未达既定55%晋级线，不追加第三组、不替换冠军。iteration 620 继续保持正式冠军，iteration 640 作为recent-75分支最佳候选保留；评估seed已恢复为200。

建立保守扩容的 stage3 分支 `stage3_depth8_from620.pt`：channels 保持32、残差块从4增至8、value head 保持8/128。初始化来源为正式冠军 iteration 620；迁移stem、前4个残差块、policy head和value head的全部权重，完整继承50,000条replay buffer，只对新增4个近恒等残差块做新初始化，optimizer重建。迁移回归测试已确认扩容前后policy/value输出在1e-6容差内一致，完整测试24 passed。配置先用旧buffer预训练500步，再训练50轮；保持固定200 simulations、recent/history 50%/50%、LR 0.0001、温度0.8→0.1、每轮10局和50 optimization steps。stage3使用自身iteration 1–50编号，metadata另行保存source iteration 620。

stage3 depth-8 首批50轮已完成，耗时约 5,802 秒（1小时36分33秒），新增500局。迁移时加载72个源张量、跳过0个，replay buffer 50,000条；扩容后参数量为579,192。首次500步buffer预训练平均 loss/policy/value 为2.6623/2.0385/0.6238。主 checkpoint 及 iteration 10/20/30/40/50 快照均已加载核对；最终metadata为stage3 iteration 50、source iteration 620、total games 500、buffer 50,000、8 blocks、LR 0.0001、reuse value head=true。iteration 1–10 平均 loss/policy/value 为2.6662/2.0448/0.6214；iteration 41–50 为2.7848/2.1022/0.6826，三项均上升，训练后期没有显示优化改善。`configs/eval.json` 已设置stage3 iteration 10/30/50使用统一200 simulations、100局及seed 200挑战stage2正式冠军iteration 620。
