# 实验性异步慢速播放

用于验证“延长动作块的播放时间，能否覆盖下一块的推理等待”。默认每块完整播放 36 步，12 Hz 对应约 3 秒；原底层 high-follow 插值仍按 200 Hz 配置运行。

这是 **stateless 异步预取**，每次模型请求仍 reset。它不是官方 KV 历史模式，也不是论文的 FDM-grounded 异步框架。`official_kv + sync` 入口和配置继续保留，两种协议不能混用。

## 使用

```bash
# 终端 1：模型环境，base/checkpoint/norm 与原推理使用相同环境变量
export LINGBOT_BASE_MODEL=/absolute/path/to/base
export LINGBOT_CHECKPOINT=/absolute/path/to/checkpoint_step_14000
PYTHON_BIN=/path/to/model/python ./start_stateless_server.sh --port 8014

# 终端 2：客户端环境。先编辑自己的相机、CAN、服务器和初始化配置。
cp integration/client_lingbot_slow_async.yaml integration/client.local.yaml
PYTHON_BIN=/path/to/client/python CLIENT_CONFIG=integration/client.local.yaml ./start_slow_client.sh --dry-run

# 准备好真机后，在交互式终端启动；按 s 开始，空格停止。
PYTHON_BIN=/path/to/client/python CLIENT_CONFIG=integration/client.local.yaml ./start_slow_client.sh
```

首次实机测试可设置 `max_publish_step: 72`。默认 null 表示持续运行，直到停止。示例相机序列号是占位符，dry-run 不验证接线和真实设备状态。

关键配置：

```yaml
inference:
  execution_mode: async
  policy_protocol: stateless
  playback_mode: slow_prefetch
  async_mode: naive
  chunk_size: 36
  execute_prefix_steps: 36
  publish_rate: 12
  observation_rate: 30
  max_prediction_wait_s: 10
  max_result_age_s: 10
  min_smooth_steps: 0
  latency_k: 0
```

`publish_rate` 支持大于 0、至多 30 Hz，可尝试 15 或 12；只改变策略动作发布时间，不改变模型训练帧率或每块动作数量。`max_result_age_s` 从该次请求准备开始计时，限制开始播放时的预测年龄；不是相机硬件时间戳校验。

## 调度与停止

1. 冷启动取得第一块后开始播放，不浪费一次 warmup 预测。
2. 播放当前块时，后台从最新可用观测请求下一块。
3. 当前块完整播放后才切换，快返回的结果不会覆盖当前块尾部。最多只有一块未来结果待播放或正在计算。
4. 下一块来不及返回时，保持当前位置并记录等待事件，不重复计数或补发过期动作；等待超时、预测过旧、错误形状或非有限值会终止本轮。
5. 停止与动作下发共用锁；清空待播结果，关闭连接以解除阻塞 RPC，迟到结果不再执行。回到初始化沿用现有客户端行为。

此模式不做 temporal_smoothing，也不进行论文 FDM 历史修正。下一块使用的是它请求时的观测，因此播放变慢可能增加控制滞后或导致接缝动作不合适。它既不把未执行动作提交为历史，也不自动缩减动作、改变归一化或重训模型。

## 已验证结果

新增 10 项测试，包括速率、单块预取上限、完整动作顺序、超时保持、推理失败、停止中断、迟到/非法结果；连同原测试共 31 项通过。

真实 step14000 模型 + 合成图像 + 内存 RobotIO，每个速率执行两块共 72 步，无硬件命令：

| 发布频率 | 36 步播放时长（理论） | 实测下一块 RPC | 额外轮间等待 |
|---|---:|---:|---:|
| 30 Hz | 1.2 秒 | 2.591 秒 | 1387.3 毫秒 |
| 15 Hz | 2.4 秒 | 2.591 秒 | 187.2 毫秒 |
| 12 Hz | 3.0 秒 | 2.605 秒 | 0.2 毫秒 |

“额外轮间等待”不包含正常的一个动作周期。完整结果见 `slow_playback_smoke.json`。上述是单次、两块、合成输入的调度测试，不是性能保证，也不是衣物任务成功率测试。GPU 竞争或更慢请求仍会等待；较低发布频率也不保证底层机械臂整段运动平滑，现有插值器仍依据路点和速度限制执行，可能出现到点保持。

复核命令：

```bash
python -m unittest discover -s integration -p 'test_*.py'
# 需要已运行的 stateless 服务；不会打开机械臂或相机：
python integration/smoke_slow_playback.py --rates 30 15 12 --port 8014
```

本次没有启动真机。优先将本功能视为时序实验；若要保留真实历史同时提前规划，需要另行实现并验证论文 FDM 条件预测和并行调度。
