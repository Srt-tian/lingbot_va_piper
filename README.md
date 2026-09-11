# LingBot-VA Piper

整理自已运行的 Piper 双臂衣物任务训练和真机推理代码。基于 [Robbyant/LingBot-VA](https://github.com/Robbyant/lingbot-va)，官方对照版本为 `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`。

本仓库包括模型/训练核心、LeRobot 数据适配和 latent 提取、Piper Runtime、官方历史推理适配、配置及测试。模型权重、训练数据、相机画面、真机运行记录和凭据不在仓库中。

## 目录

| 路径 | 内容 |
|---|---|
| `wan_va/` | 模型、训练、数据集、scheduler、配置 |
| `tools/` | 数据转换、固定数据集构建、latent 提取和验证 |
| `script/` | 已使用过的训练入口，改为显式外部路径 |
| `integration/` | Piper 模型服务、历史协议、Runtime 适配及测试 |
| `runtime/` | 从原 inference 工作树纳入的 29 个客户端依赖文件 |
| `meta/lingbot_action_norm.json` | step14000 所用数据的动作归一化；新数据需重新计算 |
| `docs/` | 训练、推理、来源和验证边界 |

## 快速开始

训练和模型服务建议使用独立 CUDA 环境；硬件客户端使用自己的环境。依赖分别见 `requirements.txt` 和 `requirements-client.txt`。官方模型环境说明见 `INSTALL.md`。FlashAttention 需与 PyTorch/CUDA 匹配；没有自动下载模型、数据或私有镜像。

训练入口（会实际训练，先按 [训练说明](docs/TRAINING.md) 准备数据）：

```bash
export DATASET_ROOT=/absolute/path/to/prepared_dataset
export MODEL_ROOT=/absolute/path/to/lingbot-va-base
export SAVE_ROOT=/absolute/path/to/checkpoints
export PY=/path/to/training/env/bin/python
bash script/entrypoint_fold_clothes_2846_fixed_20000_volcano_8gpu.sh
```

模型服务：

```bash
export LINGBOT_BASE_MODEL=/absolute/path/to/base
export LINGBOT_CHECKPOINT=/absolute/path/to/checkpoint_step_14000
export PYTHON_BIN=/path/to/model/env/bin/python
./start_history_server.sh --port 8015
```

客户端先做无硬件 dry-run：

```bash
PYTHON_BIN=/path/to/client/env/bin/python ./start_history_client.sh --dry-run
```

真机操作前必须复制配置并填写三相机序列号、CAN 和服务器地址，见 [推理说明](docs/INFERENCE.md)。默认客户端启动后等待 `s`；空格停止当前 episode 并回初始化，`q` 退出会执行配置中的归零动作。

## 可选：异步慢速播放

新增实验性 `slow_prefetch`：完整播放 36 步，默认 12 Hz，最多预取下一块。使用
`start_stateless_server.sh` 和 `start_slow_client.sh`，不连接官方历史协议服务。
真实模型内存测试中，12 Hz 的额外轮间等待约 0.2 ms；这不代表真机动作质量已验证。
配置、测试和局限见 [异步慢速播放](docs/SLOW_PLAYBACK.md)。

## 当前模式与边界

默认 `official_kv + sync`：每个 episode reset 一次，首轮执行 36 步，后续 48 步；每 3 个动作收集实际观测，再用官方 `compute_kv_cache` 回填历史。动作是左 6 关节/夹爪、右 6 关节/夹爪的 14 维绝对指令。

旧 `naive`/`temporal_smoothing` 异步代码保留作对照，但它是每请求 reset 的独立模式，不能直接与官方历史回填混用。同步模式仍有模型计算等待；本仓库不宣称无停顿或已经达到稳定的衣物任务成功率。

验证和迁移修改见 [来源与验证](docs/VALIDATION.md)。模型核心保留上游版权与 `LICENSE.txt`；纳入的 Runtime 来源见 [说明](runtime/README.md)。
