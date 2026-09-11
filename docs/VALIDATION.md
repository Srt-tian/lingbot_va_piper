# 来源与验证边界

- 官方对照：Robbyant/lingbot-va `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`。
- 整理来源：部署分支提交 `13e9537719b2e12eeae41bfeb7d1fc666b09ba07`；训练工具来自历史训练快照，原始 Git SHA 不可追溯。
- 原部署审计：23 个核心训练/推理/模型函数 AST 对照一致；scheduler 和 VAE utils 文件一致。自有数据、Piper 动作、归一化、预处理与外层服务有差异。
- 原部署验证：20 项测试；真实模型内存 IO 三轮 132 步；实机 36/48/48 三轮及随后多轮历史推理、人工停止。不是自动完成衣物任务的成功率评测。
- 打包更改：只整理代码并做路径参数化、纳入 Runtime、替换示例设备标识/测试输入、补 MODEL_ROOT override。没有启动新训练或真机测试。
- 可复核：`python integration/audit_official.py --official /path/to/official --ours . --output /tmp/official-comparison.json`。

Runtime 原文件 SHA 在 `integration/reference_manifest.json`，仍包含原工作树未提交改动的明确来源标记；不能将其称为原仓库纯净提交快照。原部署日志和相机图像没有发布。

打包后验证：21 项测试通过、29 个 Runtime 文件哈希一致、23 个官方核心函数 AST 一致、客户端 dry-run 通过；Python/shell 语法和凭据特征检查通过。结果见 package_validation.json。导入的上游文件原始空白格式保留。


## 慢速播放功能验证
新增后共 31 项测试通过，包含预取顺序、节拍、故障和停止；真实模型/内存 IO
在 30/15/12 Hz 各执行两块 72 步，均通过，无硬件命令。
结果见 slow_playback_smoke.json，边界说明见 SLOW_PLAYBACK.md。
原 package_validation.json 记录初次整理时的 21 项测试，不代表新增功能的测试总数。
