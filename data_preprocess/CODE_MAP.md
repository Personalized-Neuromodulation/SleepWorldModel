# I0002 代码使用与清理记录

整理日期：2026-09-07。判定对象：`I:\HSP\I0002-preprocess` 当前 v1.0.0 产物。

## 实际使用证据

`processed/v1.0.0/config/full_build.json` 记录 15,028 条候选记录、942 个分片，以及 core.py、pipeline.py、full.py 的 SHA256。仓库这三个文件与当前产物快照和记录摘要全部一致；`docs/I0002_PREPROCESS_PLAN.md` 也与 plan_sha256 一致。

| 代码 | 判定 | 依据 |
|---|---|---|
| i0002_v100/core.py | 保留：处理核心 | code_sha256 完全匹配 |
| i0002_v100/pipeline.py | 保留：公共流程及 pilot | code_sha256 匹配，full.py 直接导入 |
| i0002_v100/full.py | 保留：正式构建 | code_sha256 匹配，发布日志与清单对应 |
| i0002_v100/__init__.py、__main__.py | 保留：包和 CLI | 当前模块入口及导入依赖 |
| tests/test_i0002_v100.py | 保留：26 项回归测试 | 导入并覆盖当前三个实现文件 |
| scripts/scan_hsp_source_metadata.py | 保留：源扫描 | 输出 schema 与 source_scan 产物及 inventory 对应；未发现历史脚本 SHA256，不能证明运行时逐字节版本 |
| scripts/generate_hsp_source_report.py | 保留：源统计报告 | 报告结构及统计章节对应；没有运行时脚本摘要 |
| scripts/analyze_f3_m2_calibrated_ranges.py | 保留：标定复核 | 已保存报告含相应抽样章节；没有运行时脚本摘要 |

当前依赖链：CLI → full → pipeline → core；pilot 直接进入 pipeline。任务标签由 pipeline 直接读取原始 CSV，不依赖旧标签提取包。

## 已清理

- `data_preprocess/unified`、`qc_rebuild`：旧预处理算法、旧 schema 和独立入口，当前流程未引用。
- `data_preprocess/sleep_stage`、`epoch_features`：旧的独立 Parquet 标签生成流程，当前直接 CSV 流程未引用。
- `scripts/unified_baseline_20260902`：旧统一流程的对照副本。
- 旧 unified 的 audit、benchmark、resume 脚本，以及旧 qc_rebuild 审查脚本。
- `extract_pilot_flagged_epochs.py`：引用当前核心已不存在的 REASONS / WARNINGS，属于旧 pilot schema。
- 对应的旧流程测试、旧标签 requirements 和 HSP_QC_REBUILD_PLAN.md。

“未使用”限定为当前 I0002 发布主线，不表示这些历史工具从未运行。删除前已检查仓库 Python 导入关系；模型、训练、EHR、原始 H5 loader、复制源数据工具及其测试均保留。

## 保留快照的原因

产物 `config` 中的代码是发布来源凭证；`config/history/pre-task-alignment-recovery` 保存前 21 个分片恢复前的版本。两代 full_build.json 中 pipeline.py / full.py 的摘要不同，不能当成无用重复代码删除。当前三个核心文件也保持原始字节，以免破坏续跑的 SHA256 校验。

## 复现限制

发布记录引用的批准 pilot 路径当前不存在；源扫描报告中部分抽样章节对应的明细表也未出现在当前 tables 目录。此次只整理代码，没有重新清洗、补跑源扫描或更改发布状态。正式数据的使用入口为产物根 README 和发布清单。

## 回滚备份

清理后验证：全部 42 项测试通过；pilot、full 和 3 个源扫描工具的 `--help` 均正常；保留源码语法检查通过；没有残留对已删除包的 Python 引用。三个正式核心文件再次核对 SHA256，仍与发布记录一致。

本次删除的文件和整理前仓库 README 已备份并逐字节核对：`E:\Code\SleepWorldModel-code-cleanup-20260907-105823.zip`。ZIP 中 cleanup-manifest.json 列出全部删除文件。备份位于仓库外。

## 后续调整：移除 EHR

用户确认 EHR 数据尚未整理，已移除 EHR 构建与读取模块、独立测试、HSPDataset / collate 中的 EHR 接口、两个命令入口和 DuckDB 依赖。前文“EHR 保留”描述的是此前清理时的状态。原始数据目录未修改。此次删除前的代码和受影响文件备份：`E:\Code\SleepWorldModel-ehr-removal-20260907-112204.zip`。

## 后续调整：独立 dataloader 与 world_model

按用户确认完成包改名及读取层迁移：`sleep_world_model` 改为 `world_model`，旧 `data` 目录移除；`dataloader` 位于项目顶层，与模型包平级，不导入模型或预处理实现。旧 HSP 原始读取器保留在 `dataloader/readers/hsp_raw`；新 `readers/hsp.py` 读取已发布 v1.0.0 分片。

训练和调试默认使用正式分片，并从数据元信息构造 15 路、200 Hz 的输入配置。当前六模态模型使用保守窗口 QC 策略，细节见项目 README；Reader 保留全部逐 epoch 质量信息和按需加载的任务标签。

验证：52 项测试通过；两个真实分片（00000、00021）的信号、QC、三个 task 只读校验及模型前向/反向通过；冻结预处理代码 SHA256 保持不变。SleepWM 环境已刷新本项目的 editable 安装，在仓库外验证两个顶层包与三个已安装命令入口均可使用。未升级第三方依赖。

迁移前源码与配置备份：`E:\Code\SleepWorldModel-dataloader-migration-20260907-114615.zip`。
