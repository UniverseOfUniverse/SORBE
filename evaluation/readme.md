## Project Overview

This project provides a comprehensive pipeline for evaluating Large Multimodal Models (LMMs) on biomedical datasets, covering complexity labeling, response generation, and quantitative scoring.

------

### 1. `dok_judge-current.ipynb`

**Function: Dataset Difficulty Classification**

- Implements the **Depth of Knowledge (DOK)** framework (Levels 1–4) to categorize the cognitive complexity of biomedical VQA items.
- Automatically assigns difficulty levels based on whether a question requires simple recall, conceptual application, or strategic reasoning.

### 2. `evaluation-current.ipynb`

**Function: Model Inference & Response Generation**

- Handles large-scale inference by calling various LLM/LMM APIs (e.g., Qwen-VL, Gemini) in both synchronous and asynchronous modes.
- Deploys models on the dataset to collect raw outputs and reasoning processes for further evaluation.

### 3. `analyze_results.py`

**Function: Quantitative Scoring & Statistical Analysis**

- Parses model judgments to calculate performance metrics:
  - **Conclusion Score:** Accuracy of the final answer.
  - **Process Score:** Logical correctness of the reasoning steps.
  - **LCR (Harmonic Mean):** A weighted balance between reasoning and conclusion.
- Identifies specific failure modes, such as visual recognition errors or interpretation mistakes.

------

