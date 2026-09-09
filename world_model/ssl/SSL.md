# 最小 SSL 与 SIGReg

## Review 与改写范围

2026-09-07 按“简化整个 SSL，保留最小训练闭环”重写。SSL 核心从 8 个 Python 文件、1,238 行缩减为 5 个文件、219 行（含注释与空行）。

| 旧实现的问题 | 本次处理 |
|---|---|
| SSL 核心有 8 个 Python 文件、1,238 行，包含 fast / epoch / context / cross-modal 多层预测和遮挡配置，超出当前基线需要 | 删除层级预测、Transformer、专用 masking 和 tensor 包装，保留 CNN、投影头和两个损失分量 |
| SIGReg 在三角函数运算后乘 mask；无效行含 NaN 时仍返回 NaN，已复现 | 在所有运算前用布尔索引移除无效行；有效行非有限值直接报错 |
| 有效样本不足两个时返回零，可能静默关闭正则项 | 明确报错，提示增加 batch 或窗口长度 |
| 固定六模态接口，QC 排除了窗口内出现过坏 epoch 的整个通道 | 保留读取器的模态分组，对每个 epoch 的每个通道应用 QC |

旧 SIGReg 的特征函数思路保留；本次主要简化其实现和调用路径。改写前源码备份为 `E:\Code\SleepWorldModel-ssl-rewrite-20260907-143810.zip`，历史模型权重保留。

## 文件与数据流

| 文件 | 职责 |
|---|---|
| `config.py` | 通道、采样率、网络宽度和裁剪比例 |
| `model.py` | 每模态两层 Conv1d、全局池化、融合层和共享投影头 |
| `sigreg.py` | 随机投影、经验特征函数、标准高斯目标和数值积分 |
| `losses.py` | 两视图 MSE 与 SIGReg 加权求和 |
| `__init__.py` | 公共导出 |

独立 `dataloader` → `training.batch_adapter.prepare_batch` → `SSLModel` → `SSLLoss` → backward → AdamW → checkpoint。

输入为各模态的 `[B,E,C,S]` 信号和 `[B,E,C]` 有效通道 mask。适配器合并通道存在性、通道 mask、逐 epoch QC 和 padding mask，先把无效值置零，再检查有效信号是否有限。

模型内部进行 `sign(x) * log1p(abs(x))` 幅度压缩，避免不同物理单位的数值尺度直接主导卷积；这是当前基线的固定输入变换，不是经验证的最优归一化方案。原始物理值和 QC 文件不被改写。

训练时抽取两个随机裁剪位置，默认每个视图保留 epoch 的 80%。同一视图使用统一的相对位置对齐各模态；该位置在 batch 内共享。没有有效通道的模态，其编码特征置零；没有任何有效模态的 epoch 不参与损失。

投影输出为 `[B,E,projection_dim]`。`model.encode(batch)` 默认编码完整 epoch，返回 `[B,E,embedding_dim]` 表示和 `[B,E]` 有效 mask，适合下游使用。

## 目标函数

设两个视图的有效 epoch 投影为 `z1`、`z2`，形状均为 `[N,D]`：

```text
invariance = mean((z1 - z2)²)
regularization = (SIGReg(z1) + SIGReg(z2)) / 2
loss = (1 - λ) * invariance + λ * regularization
```

默认 λ=0.05，两条分支都参与反向传播。MSE 使用直接的两视图差，因此权重对应上面明示的公式。

SIGReg 参考 [LeJEPA 官方最小实现](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md) 和 [论文](https://arxiv.org/abs/2511.08544)：

1. 每次调用采样 256 个随机单位方向，将 `[N,D]` 投影成 `[N,256]`。
2. 计算经验特征函数的实部 `mean(cos(t*x))` 和虚部 `mean(sin(t*x))`。
3. 与标准高斯特征函数 `exp(-t²/2)` 比较，计算平方误差，再乘同一高斯权重。
4. 在 `[0,3]` 的 17 个点上使用梯形积分，乘 2 恢复对称区间，再乘样本数 N、对方向取平均。

正则项固定使用 FP32；有效样本不足两个或包含 NaN/Inf 时报错。没有额外的样本截断、分组、top-k 惩罚或分布式汇总。投影中间张量随 `N × 方向数 × 积分点数` 增长。

## 训练边界

这是逐 epoch 表示学习基线；`context-epochs` 仅控制读取窗口，当前没有跨 epoch 时序预测。相邻 epoch 也不能当作独立被试，实际表示质量需要下游任务验证。

训练循环保留 AdamW、固定学习率、梯度裁剪、可选 W&B 和训练结束时保存权重。使用单设备本地 batch；不包含断点续训或分布式训练入口。默认 W&B 关闭，默认 checkpoint 为 `artifacts/checkpoints/ssl_minimal.pt`。

checkpoint 使用 schema 3，标记 `architecture=minimal-sigreg-v1`；模型配置和读取配置一并保存。旧层级模型权重不兼容新架构。运行命令与加载示例统一放在项目根 README 中。

## 验证

自动测试覆盖独立 NumPy/SciPy 复数特征函数数值对照、坍塌与非高斯分布识别、无效 NaN/Inf 排除、FP32 正则计算、两视图梯度、逐 epoch QC、padding、训练 CLI、checkpoint 重载和调试入口。

本次验证结果：

- 全套 58 项测试通过；修改的 Python 文件通过 Ruff 检查和格式化。
- 从正式发布的 shard 00000 与 00021 各读取两个 epoch，覆盖 pilot.2 和 full.1 两种 schema。在 RTX 4090 D 上对该 batch 执行两步 CUDA BF16 训练，损失和梯度有限、参数发生更新。
- 保存后重载模型，完整 epoch 表示形状为 `[2,2,128]`，与保存前逐值一致。该检查只验证训练闭环，不代表训练收敛或下游效果。
- 仓库外的已安装训练命令与合成数据调试命令运行通过。
- 冻结预处理的 core.py、pipeline.py、full.py SHA256 均与正式发布配置一致。正式信号文件保持只读。
