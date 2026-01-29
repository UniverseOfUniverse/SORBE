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

---

### Datasets

---



