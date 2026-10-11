# ConnectFour3D AlphaZero

一个用于训练 AI 玩 **5×5×5 立体四子棋** 的紧凑型强化学习项目，整体采用类似 AlphaZero 的架构：

- 3D 卷积残差策略价值网络
- MCTS（蒙特卡洛树搜索）
- 自博弈生成训练数据
- Replay Buffer 与 D4 棋盘对称增强
- checkpoint 续训与模型竞技评估
- Tkinter 人机对弈界面

## 当前模型

当前 UI 默认冠军为 **Stage3 iteration 130**：

```text
saved_models/deployment/stage3_depth8_champion_iter_000130.pt
```

网络使用 32 channels、8 个残差块，训练和主要评估均使用 200 次 MCTS simulations。

## 环境安装

推荐使用 Python 3.11，并安装以下依赖：

```bash
pip install numpy torch gymnasium matplotlib tqdm pytest
```

如需 GPU 训练，请根据本机 CUDA 环境安装对应版本的 PyTorch。

## 启动 UI

```bash
python run_ui.py
```

UI 支持：

- 选择 AlphaZero、历史冠军或 Alpha-Beta 对手
- 调整 MCTS simulations
- 实时显示 value network 局面估值与曲线
- AI 推荐落子
- 悔棋与棋局记录

## 继续训练

Stage3 训练参数位于：

```text
configs/train_stage3_depth8_from620.json
```

启动训练：

```bash
python run_stage3_depth8_training.py
```

程序会从 `saved_models/stage3_depth8_from620.pt` 恢复网络、优化器、Replay Buffer 和训练进度。

## 模型评估

评估参数位于 `configs/eval.json`：

```bash
python run_evaluation.py
```

评估支持 checkpoint 之间的竞技对比、成对随机开局、交换先后手以及实时进度显示。

## 测试

```bash
python -m pytest -q
```

## 项目结构

```text
agents/          AI Agent
environments/    5×5×5 四子棋环境
mcts/            MCTS 搜索
models/          策略价值网络
self_play/       自博弈与 Replay Buffer
training/        网络训练与 checkpoint
evaluation/      模型评估
ui/              桌面人机对弈界面
configs/         训练与评估配置
records/         架构、实验和棋局记录
saved_models/    续训 checkpoint 与部署模型
```

更多实现约定与训练记录可参考：

- `records/project_architecture.md`
- `records/training_log.md`
- `records/stage3_depth8_evaluation.md`

