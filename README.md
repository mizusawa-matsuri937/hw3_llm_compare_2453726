# 大语言模型部署体验实验

## 一、项目简介

本项目为人工智能导论第三次课程作业，实验主题为“大语言模型部署体验”。

实验基于 ModelScope 魔搭平台，在 CPU Notebook 环境下部署并测试开源大语言模型。实验中选取 `Qwen-7B-Chat` 和 `ChatGLM3-6B` 两个中文大语言模型，对其进行统一问答测试，并从中文语义理解、多层逻辑推理、指代消解、回答完整性和 CPU 推理效率等方面进行横向对比分析。

## 二、基本信息

- 姓名：付林轩
- 学号：2453726
- 平台：ModelScope 魔搭平台
- 环境：CPU Notebook
- Python 版本：Python 3.11.11
- PyTorch 版本：2.3.0+cpu
- Transformers 版本：4.33.3
- ModelScope 版本：1.9.5

## 三、测试模型

本实验部署并测试了以下模型：

1. `Qwen-7B-Chat`
2. `ChatGLM3-6B`

说明：由于模型文件体积较大，本仓库不直接上传模型权重文件，只保存实验脚本、测试结果、运行截图和实验报告。

## 四、仓库结构

```text
hw3_llm_compare/
├── README.md
├── scripts/
│   ├── run_qwen_cpu.py
│   └── run_chatglm3_cpu.py
├── results/
│   ├── result_qwen.txt
│   └── result_chatglm3.txt
├── screenshots/
│   └── 实验过程截图
└── report/
    └── hw3_2453726_付林轩.docx
```

## 五、实验问题设计

实验使用统一问题集测试不同模型，主要包括：

- 中文语义歧义理解
- 双关句理解
- 多层嵌套逻辑推理
- 指代消解
- 多义词解释
- 大语言模型基础概念解释
- 简单逻辑推理判断

## 六、运行方式

在 ModelScope CPU Notebook 环境中，进入实验目录后运行：

```bash
python scripts/run_qwen_cpu.py
python scripts/run_chatglm3_cpu.py
```

如果脚本已经放在当前目录下，也可以直接运行：

```bash
python run_qwen_cpu.py
python run_chatglm3_cpu.py
```

## 七、实验结果

详细测试结果保存在 `results/` 文件夹中。

- `result_qwen.txt`：`Qwen-7B-Chat` 的问答测试结果
- `result_chatglm3.txt`：`ChatGLM3-6B` 的问答测试结果

运行截图保存在 `screenshots/` 文件夹中。

## 八、实验结论

通过测试可以看出，两个模型都能够完成基本中文问答和常识推理任务，但在复杂歧义句、多层逻辑关系和指代消解问题上存在差异。

`Qwen-7B-Chat` 在部分问题上表达更详细，但 CPU 推理耗时较长；`ChatGLM3-6B` 在回答速度和简洁性方面表现较好，但个别歧义理解题存在解释不够准确的情况。

本实验表明，在 CPU 环境中部署 7B 级别大语言模型具有较高的资源消耗，模型选择需要综合考虑回答质量、部署成本和推理效率。
