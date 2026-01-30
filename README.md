Benchmarking the Scientific Mind: Toward Evaluation of Complex-Reasoning Biomedical VQA
---



### Introduction

---

Despite progress of Multimodal Large Language Models (MLLMs) in biomedical visual question answering (VQA), existing benchmarks provide limited assessment of their scientific reasoning capabilities. Most datasets adopt single-image question construction and outcome-oriented evaluation, where correctness is judged by answer plausibility rather than alignment with experimental evidence. Such formulations fail to capture the evidence-constrained, multi-step nature of biomedical reasoning, and obscure whether models can derive conclusions through causal interpretation of experimental observations.

To address these critical gaps in reasoning evaluation, we propose a principled benchmark construction framework that reconstructs scientific reasoning paths directly from biomedical literature. By jointly modeling clusters of experimentally related images together with their captions and context, the framework generates tightly coupled question–reasoning–answer triples that require multi-image integration and explicit evidence-driven inference. Based on this framework, we introduce SORBE - Scientific Observation Reasoning for Biomedical Evaluation, a large-scale multi-image biomedical VQA benchmark designed to evaluate evidence alignment and multi-step experimental reasoning. Under a process-oriented evaluation metric, state-of-the-art biomedical-specialized MLLMs exhibit substantial performance degradation, revealing systematic limitations in evidence grounding and causal reasoning that are not reflected by existing benchmarks.

![Figure1](figure/Figure1.png)
Figure 1. (A) Comparison between the typical biomedical VQA Benchmark and our SORBE. Our process-oriented scoring evaluates
verifiable reasoning structure rather than subjective explanation quality. (B) Scores of the three models, GPT-4o, Lingshu-32B, and Hulu-32B, on PMC-VQA, MMMU-Med, and our SORBE, respectively.


![Figure2](figure/Figure2.png)
Figure 2. Overview of Benchmark Construction Framework. (A) Contextual Knowledge Extraction, which distills structured
experimental metadata from unstructured biomedical literature; (B) Reasoning Path Construction, which extracts visual evidence and
constructs logic chains from images and contextual knowledge; and (C) Question Generation & Filtering, which produces open-ended
questions and filters out low-quality questions.

---

### Installation

---

Install Python 3.9.25

```
conda env create -f med_qa_env.yml
conda activate med_qa_env
```

Our framework operates on a plug-and-play architecture, allowing for the flexible replacement of any  models. By default, the system is configured as follows: for image-to-text generation via the VLM_PROMPT_TEMPLATE, it utilizes a suite of vision-language models, specifically Qwen3-VL-235B-A22B-Instruct, Lingshu-32B, Hulu-Med-32B, and Fleming-VL-38B. For all other functional steps, Qwen3-235B-A22B-Instruct serves as the primary text-based backbone.

如果要使用其他模型，请更改每个代码如下的设置：

文本模型的设置，本文默认使用Qwen3-235B-A22B-Instruct

```
local_text_api_key = "xxx"
local_text_model = "xxx"

local_text_client = AsyncOpenAI(
    api_key=local_text_api_key,
    base_url="xxx", 
    timeout=120.0
)
```

视觉模型设置，本文使用Qwen3-VL-235B-A22B-Instruct, Lingshu-32B, Hulu-Med-32B, and Fleming-VL-38B

```
local_vl_api_key = "xxx"
local_vl_model = "xxx"

local_vl_client = AsyncOpenAI(
    api_key=local_vl_api_key,
    base_url="xxx", 
    timeout=120.0
)
```

---

### Datasets

---

You can use the `Data\data_final.json` file in the dataset to download a demo of the processed SORBE dataset, which contains 200 data points for you to run subsequent QC, evaluation, and testing code.

---

### Usage

---

Quick Start：你可以使用SORBE\qa_generation中的xxx.ipynb进行qa生成中的每步骤分开实现这个流程中的任何一个流水线，方便您查看每个流水线后的结果。

All run: 要遍历全量数据，需要使用SORBE\qa_generation中的xxxx.py，来进行端到端的生成数据。

---

### Evaluation

---

需要使用SORBE\evaluation中的xxx.py 和xxx.py ，前者可以获得不同模型答案。后者则可以根据逻辑链对不同模型进行测评他们的得分。

---

评测结果



---

Licese



引用



致谢



