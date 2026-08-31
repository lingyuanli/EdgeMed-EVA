# Survey 来源台账

访问日期：2026-08-31。优先列一手论文、官方仓库和官方模型/硬件文档。

| ID | 来源 | 类型 | 本轮用途 |
|---|---|---|---|
| S01 | [Med-CMR official repository](https://github.com/LsmnBmnc/Med-CMR)；本地所附论文 PDF | benchmark/全文深读 | 数据规模、七维度、baseline、公开可得性 |
| S02 | [MMedAgent](https://aclanthology.org/2024.findings-emnlp.510.pdf) | EMNLP Findings/全文深读 | 多医学工具动作规划与局限 |
| S03 | [DeepEyes](https://arxiv.org/abs/2505.14362)；[official code](https://github.com/Visual-Agent/DeepEyes) | 论文/代码/全文深读 | crop-and-zoom RL 与数据筛选 |
| S04 | [What Does Vision Tool-use RL Really Learn?](https://arxiv.org/abs/2602.01334)；[official code](https://github.com/GAIR-NLP/Med) | 论文/代码/全文深读 | intrinsic learning、Call Gain/Harm 分解 |
| S05 | [MAIRA-2](https://arxiv.org/abs/2406.04449) | 论文/全文深读 | grounded report、多图输入、证据指标 |
| S06 | [MMedPO](https://arxiv.org/abs/2412.06141)；[official code](https://github.com/aiming-lab/MMedPO) | 论文/代码/全文深读 | 医学视觉偏好、负例构造 |
| S07 | [ChestX-Reasoner](https://arxiv.org/abs/2504.20930)；[journal version](https://www.nature.com/articles/s43856-026-01654-y) | 论文/全文深读 | 答案 SFT、过程奖励与训练顺序 |
| S08 | [MedTrinity-25M](https://arxiv.org/abs/2408.02900) | 数据/论文 | 大规模多模态医学 grounding 数据 |
| S09 | [MedXpertQA](https://arxiv.org/abs/2501.18362) | benchmark | 医学专家级文本/多模态推理 |
| S10 | [GMAI-MMBench](https://proceedings.neurips.cc/paper_files/paper/2024/file/ab7e02fd60e47e2a379d567f6b54f04e-Paper-Datasets_and_Benchmarks_Track.pdf) | NeurIPS D&B | 多模态医学评测景观 |
| S11 | [MedHELM](https://medhelm.org/) | benchmark platform | 多维透明评测思路 |
| S12 | [VILA-M3](https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html) | CVPR | 医学专家知识与视觉语言模型 |
| S13 | [Lingshu](https://arxiv.org/abs/2506.07044) | 论文 | 通用医学多模态 foundation model 参考 |
| S14 | [MedGemma official repository](https://github.com/google-health/medgemma) | 模型/代码 | 医学开源 baseline 候选 |
| S15 | [VoxelPrompt](https://arxiv.org/abs/2410.08397) | 论文 | 3D 医学图像 grounding 候选方向 |
| S16 | [MedRAX](https://arxiv.org/abs/2502.02673) | 论文/代码入口 | 放射学多 Agent/工具系统边界 |
| S17 | [MultiMedEval](https://proceedings.mlr.press/v250/royer24a.html) | 评测工具 | 医学多模态统一评测 |
| S18 | [HEAL-MedVQA](https://www.ijcai.org/proceedings/2025/0853.pdf) | IJCAI | localize-before-answering 路线 |
| S19 | [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B)；[ms-swift official repository](https://github.com/modelscope/ms-swift) | 模型/训练生态 | backbone、原生 VL、工具与 QLoRA 实现入口 |
| S20 | [NVIDIA CUDA BF16 support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)；[Tesla V100](https://www.nvidia.com/en-gb/data-center/tesla-v100/) | 官方硬件文档 | V100 FP16/BF16 兼容性边界 |

## 使用规则

- `paper-reported` 数值必须保留来源、模型版本和评测协议；
- arXiv 与正式发表版并存时，优先用正式版，方法版本不一致则分别记录；
- GitHub README 只能支持公开可得性/使用方式，不能代替论文实验；
- 新来源先进入本台账和对应 paper card，再改变方法设计。
