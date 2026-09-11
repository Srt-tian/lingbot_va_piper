# 真机推理

## 环境与模型

服务器：安装 `requirements.txt`，准备 LingBot-VA base 的 transformer、vae、text_encoder、tokenizer 等目录，以及微调 checkpoint。默认搜索仓库 `models/base` 和 `models/step14000`；也可使用 `LINGBOT_BASE_MODEL`、`LINGBOT_CHECKPOINT`。非 14000 权重同时设置 `LINGBOT_CHECKPOINT_STEP`，该值只是 metadata 标记，不能验证权重版本。`LINGBOT_NORM_PATH` 可选择与训练数据匹配的 norm 文件，默认 `meta/lingbot_action_norm.json`。

客户端：安装 `requirements-client.txt`，需要 Piper SDK、RealSense 和 CAN 系统支持。`runtime/` 是已验证的原 inference 源码快照，不再要求 `/root/wudi/inference`。用 `INFERENCE_REPO_ROOT` 或 `--reference-root` 切回外部仓库时，29 个文件的 SHA256 必须与 manifest 一致。

```bash
PYTHON_BIN=/path/to/server/python ./start_history_server.sh --port 8015
```

服务目前为单历史会话，无鉴权；使用可信内网或 SSH 转发，不能把端口直接暴露到公网。服务启动会加载模型，所需空闲显存取决于共用 GPU 的其他任务。历史实机长轮曾因另一个进程额外占约 9.7 GiB 导致 OOM；`healthz` 可访问不代表失败会话可继续，应重启服务并重新 reset。

## 客户端配置

```bash
cp integration/client_lingbot_history.yaml integration/client.local.yaml
# 编辑 client.local.yaml：相机 serial、CAN 对应关系、server.host/port、初始化姿态
PYTHON_BIN=/path/to/client/python CLIENT_CONFIG=integration/client.local.yaml ./start_history_client.sh --dry-run
# 真机已上电并准备好以后，从交互式终端运行：
PYTHON_BIN=/path/to/client/python CLIENT_CONFIG=integration/client.local.yaml ./start_history_client.sh
```

dry-run 只核验软件配置，不开启硬件，也不证明相机/机械臂接线正确。共享配置的相机序列号是占位符，必须替换。原物理相机键映射不能从 front/left/right 名字直接推断，请核对三相机实际画面。

启动客户端会连接设备并到初始化姿态，然后等待 `s`。空格结束 episode，清队列并回初始化。`q` 退出会按配置执行归零动作。初次验证可把 `max_publish_step` 改为 132（36+48+48），默认 null 表示不自动停止。

## 历史时序

每 episode reset 一次。首次模型返回 `[14,4,12]`，第一个 12 步条件块不执行，因此首轮 36，后续 48。每 3 个动作取得一组三相机观测，首轮 12 组、后续 16 组；完成后回填全部原始动作块和实际观测，frame start 0→4→8。当前状态向量不作为额外模型条件，回填接口的 `state` 指动作块。

新一轮 reset 清除上次历史；停止中途不把未执行尾部当作完成历史。回填失败停止本轮，不自动忽略错误继续执行。由于是同步完整 chunk，30 Hz 只描述轮内发布频率；不能用其宣称全程连续 30 Hz。历史测得模型约 2.6 秒、回填约 0.5 秒，48 步执行约 1.6 秒。

## 测试

```bash
python -m unittest discover -s integration -p 'test_*.py'
# 已启动模型服务时，可运行纯内存 IO 的真实模型 smoke（不连接机械臂）：
python integration/smoke_official.py
```

打包版本的 smoke 使用合成图像，只能验证协议、输出形状与历史流程，不能评价衣服任务效果。旧 stateless/异步配置只用于对照，需要配套的 `serve_lingbot.py` 服务，不能连接 `official_kv` 服务。


## 可选慢速播放
`slow_prefetch` 提供无历史服务上的完整块预取、可调发布频率和等待超时处理，
详见 [SLOW_PLAYBACK.md](SLOW_PLAYBACK.md)。它不改变本页的官方 KV 同步流程，
也不是论文的 FDM-grounded 异步算法。
