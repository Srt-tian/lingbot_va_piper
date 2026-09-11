# 训练与数据准备

## 数据约定

原始动作顺序为 `[left_joint1..6, left_gripper, right_joint1..6, right_gripper]`，14 维，关节弧度、夹爪位置；映射到 30 维通道 `[14,15,16,17,18,19,28,21,22,23,24,25,26,29]`。相机键为 `observation.images.top_head/hand_left/hand_right`。原动作 30 Hz，图像抽样 10 Hz，VAE 时域压缩后每 latent 帧对应 12 个动作。RGB uint8 使用 OpenCV INTER_AREA 缩放到 256×256。

历史 step14000 来自 2846 段固定衣物数据，不能用任意新数据加同一份 norm 冒充复现。源码原始训练机没有 Git 元数据；本仓库记录可核对的部署来源，不编造训练源提交。

## 工具顺序

在保存数据的计算服务器执行，避免将数据或权重下载到控制工作站：

```bash
export PYTHONPATH="$PWD:$PWD/wan_va${PYTHONPATH:+:$PYTHONPATH}"
python tools/adapt_fold_lerobot_for_lingbot.py --src /data/lerobot --dst /data/adapted
python tools/extract_fold_latents.py --dataset /data/adapted --model /models/base   --target-fps 10 --ori-fps 30 --height 256 --width 256 --device cuda
```

上述路径均是示例。使用 `--help` 查看分段与分卡提取参数。提取脚本同时生成零 empty embedding，训练和在线 negative conditioning 与此对齐。

`tools/build_fixed_fold_dataset.py` 是历史特定数据修复工具：输入是已筛选的 2852 段（长度上限 1800 帧）数据，排除固定 6 个 episode ID，得到 2846 段并重算 q01/q99；它本身不负责生成前置的 2852 段筛选结果。不要在其他数据集上盲用这些 ID。对新数据应制定自己的筛选规则并计算匹配的 norm。该工具使用数据/视频/latent 符号链接，必须保留源目录。

```bash
python tools/build_fixed_fold_dataset.py --src /data/filtered_2852 --dst /data/fixed_2846
export DATASET_ROOT=/data/fixed_2846
python tools/validate_fold_lingbot_dataset.py --config-name fold_clothes_train --expected-len 2846
```

## 训练入口

```bash
export DATASET_ROOT=/data/fixed_2846
export MODEL_ROOT=/models/base
export SAVE_ROOT=/checkpoints/piper_run
export PY=/path/to/env/bin/python
export WANDB_DISABLED=true
bash script/entrypoint_fold_clothes_2846_fixed_20000_volcano_8gpu.sh
```

默认 8 GPU、每卡 batch 1、梯度累计 2、学习率 2.5e-5、20000 步、每 2000 步保存。step14000 是此设置下选取的历史 checkpoint，不是“训练目标 14000 步”。`NGPU`、`LINGBOT_LR`、`LINGBOT_NUM_STEPS` 等可通过环境变量修改。`LINGBOT_VALIDATE_ONLY=1` 仅做数据验证，不训练，但入口仍检查 GPU。

W&B 通过环境变量注入 `WANDB_API_KEY`、`WANDB_TEAM_NAME`、`WANDB_PROJECT`、`WANDB_NAME`；不把密钥写进脚本或提交。入口不提交 EIP 任务。代码中的其他 demo/历史配置保留参考，推荐使用上述入口覆盖数据、模型和输出路径。

本次整理补上训练 `MODEL_ROOT` 的实际读取，未改变加噪、loss、模型 forward 或 KV 核心逻辑。多片段 latent 匹配修复已经包含在数据集代码中；历史 2846 段核对全部精确匹配。
