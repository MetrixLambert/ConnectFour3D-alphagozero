# Stage3 depth-8 评估记录

## 首批 50 iterations

- 候选模型：`stage3_depth8_from620_iter_000010.pt`、`000030.pt`、`000050.pt`
- 固定对手：stage2 正式冠军 `stage2_sim200_temp08_lr1e4_from590_iter_000620.pt`
- 双方搜索预算：200 simulations
- 每组：100 局，2 手成对随机开局

seed 200 首轮结果：

| checkpoint | 胜/和/负 | 胜率 |
| --- | ---: | ---: |
| stage3 iteration 10 | 50/0/50 | 50% |
| stage3 iteration 30 | 54/0/46 | 54% |
| stage3 iteration 50 | 51/0/49 | 51% |

iteration 30 最优且接近晋级线，因此追加 seed 201 的 100 局确认赛，结果为 60/0/40（60%）。两组累计为 114/0/86，即 57%，超过既定 55% 晋级线。

iteration 30 与 iteration 50 在共同对手上的首轮成绩相差 3 局。最终保留经过独立 seed 确认、累计胜率 57% 的 `stage3_depth8_from620_iter_000030.pt` 为正式冠军；iteration 50 仅作为后续实验的训练起点。评估 seed 已恢复为 200，固定对手保持 stage3 iteration 30。

后续实验从 iteration 50 完整快照继续训练 50 轮，用于确认 30→50 阶段观察到的回落是否会延续。为增大参数变化，本轮学习率由 `0.0001` 提高至项目此前使用过的 `0.0003`。误启动的低学习率续训在 iteration 84 后停止，其主 checkpoint 已恢复为 iteration 50，相关 60/70/80 快照已清理，不纳入该实验。

## iteration 50→100 高学习率续训

从 iteration 50 使用 `learning_rate=0.0003` 继续训练 50 轮，最终完整状态为 iteration 100、total games 1000、buffer 50,000。训练因单次执行窗口分为 50→85 与 85→100 两段，但始终从完整 checkpoint 恢复，学习率和 optimizer 状态连续。

对 iteration 30 冠军的 seed 200 首轮结果：iteration 60 为 51/0/49，iteration 80 为 55/0/45，iteration 100 为 55/0/45。对 iteration 80/100 追加 seed 201 确认赛，结果分别为 61/0/39 与 57/0/43。两组累计：iteration 80 为 116/0/84（58%），iteration 100 为 112/0/88（56%）。两者均超过 55% 晋级线，说明 iteration 50 的回落不是持续退化；提高到 `0.0003` 后继续训练产生了可复现的棋力收益。

iteration 80 与 100 的累计成绩仅差 2 个百分点，按“候选接近时优先选择训练更充分的较新 checkpoint”规则，`stage3_depth8_from620_iter_000100.pt` 晋升为正式冠军。评估 seed 恢复为 200，固定对手更新为 iteration 100；iteration 80 保留为默认复测候选。UI 尚未同步，等待单独部署指令。

## iteration 100→150 续训

保持 LR 0.0003 与其余参数不变训练 50 轮。seed 200 下，iteration 110/130/150 对 iteration 100 分别为 42/0/58、57/0/43、52/0/48。iteration 130 的 seed 201 确认赛为 58/0/42，两组累计 115/0/85（57.5%），因此 iteration 130 晋升为正式冠军；iteration 150 未晋级。评估 seed 已恢复为 200，训练默认批次数恢复为 50。
