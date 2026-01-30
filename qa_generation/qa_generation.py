import asyncio
import json
import os
import base64
import re
import time
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI, APIConnectionError, InternalServerError

# ================= CONFIGURATION =================
DASH_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "your_key_here"
LOCAL_VL_API_KEY = "xxx"
LOCAL_VL_BASE_URL = "xxx"
LOCAL_TEXT_API_KEY = "xxx"
LOCAL_TEXT_BASE_URL = "xxx"

# 模型名称
VL_MODEL_NAME = "qwen3_vl_235b_instruct"
TEXT_MODEL_NAME = "qwen3_235b_instruct"

# 初始化客户端
local_vl_client = AsyncOpenAI(api_key=LOCAL_VL_API_KEY,
                              base_url=LOCAL_VL_BASE_URL,
                              timeout=120.0)

local_text_client = AsyncOpenAI(api_key=LOCAL_TEXT_API_KEY,
                                base_url=LOCAL_TEXT_BASE_URL,
                                timeout=120.0)

# ================= PROMPT DEFINITIONS (PASTE YOUR PROMPTS HERE) =================

BIOMED_CHECK_PROMPT = """
You are a data classifier. Your task is to determine if the provided text context describes Biomedical, Medical, or Clinical content (e.g., pathology, anatomy, cell biology, medical imaging, clinical reports).

Input Context:
{context}

Output Requirement:
Return ONLY a JSON object with a single boolean field `is_biomedical`.
Example: {{"is_biomedical": true}} or {{"is_biomedical": false}}
"""

BACKGROUND_DISTILLATION_PROMPT_TEMPLATE = """
As an expert biomedically scientific editor, your task is to distill the provided [Background] text into a concise, focusing on biomedical entity information.

Input:
[Background]:
{back_info}

The distilled summary MUST meet the following requirements:
1.  Length: a reasonable summary between 100 and 200 words.
2.  Focus: Focus on explaining the core scientific problems within the background context, including the important knowledge related to them..
3.  Style: Use formal, clear, and objective scientific language.

Output:
"""

KEYWORD_Category_PROMPT_TEMPLATE = """
You are a top-tier biomedical research analyst, skilled at structured information extraction and thematic classification. Your task is to perform a two-step analysis on the provided [Context] and [Image_caption].

I. INPUT DATA

[Context]:
{context}
*(Note: The [Context] may contain `[Image]` tokens indicating image positions. You CANNOT see these images; you MUST rely *only* on the Observation for all visual details.)*

[Image_caption]:
{image_caption}

II. STEP 1: THEMATIC CLASSIFICATION
Analyze the content and select **ONE** Main Category (1-4) that best describes the core research domain. Use the professional framework below.

CLASSIFICATION FRAMEWORK:
- Basic Medical Science :
  Focuses on the fundamental mechanisms of life and disease.
  (Keywords: Molecular biology, genetics, biochemistry, immunology, physiology, anatomy, neurosciences, cellular pathways, pathogenesis models).
- Clinical Medicine :
  Focuses on the diagnosis, treatment, and management of human diseases in patients.
  (Keywords: Specific diseases (e.g., Heart Disease, Endocarditis), surgical procedures, patient case studies, treatment outcomes, clinical neurology, ophthalmology, urology, orthopaedics).
- Diagnostics & Laboratory Medicine :
  Focuses on the methods and technologies used to detect and diagnose diseases.
  (Keywords: Pathology, histopathology, cytopathology, medical imaging (Radiology, MRI, CT), biomarkers, lab tests, assay development, neuropathology, forensic analysis, electrocardiorgraphy).
- Pharmacy & Therapeutics :
  Focuses on the discovery, development, and application of drugs.
  (Keywords: Pharmacology, drug synthesis, medicinal chemistry, drug targets, therapeutic strategies, drug resistance, clinical trials for drugs, pharmaceutical sciences).

III. STEP 2: THEME-GUIDED KEYWORD EXTRACTION
Based on the [Context] and the provided Observation and Interpretation as well as the classification result from STEP 1, extract a list of 10-15 highly specific biological or medical keywords.
- **CRITICAL:** Ensure keywords are directly relevant to the selected Main Category.
- **Focus on:** Specific protein/gene names, cell types/morphologies, disease names, diagnostic criteria (e.g., grading), specific drugs, or key experimental findings.
- **Avoid:** Generic words or phrases.



IV. REQUIRED OUTPUT FORMAT
Output *only* the result in the following exact structure:
[Main Category Name]: keyword1, keyword2, keyword3, ..., keyword15
"""

VLM_PROMPT_TEMPLATE = """
You are an expert biologist and biomedical researcher. You will be given a image and [Image Captions].Your task is to describe the visual content of the image. 

# Input Data
[Image Caption]: 
{caption}

# Task
Provide A detailed description of the visual features present in the image,  grounded in the [Image Captions], Avoid using any conclusive statements. Focusing on the observation of visual features in biomedical images.

# Constraints
- Do NOT output a list. 
- Do NOT mention "Image 1" or other image indices.
- Output ONLY the description paragraph.
"""

CONSENSUS_PROMPT_TEMPLATE = """
You are a senior biomedical image analyst. You are a senior biomedical image analyst. You will receive observations from four different biomedical experts regarding the same biomedical image.These observations may include their interpretations or inferences based on the image, which you should disregard.

#Task:
You should limit yourself to purely visual descriptions, avoid adding any explanatory logic, and extract the common visual information from these observations, while avoiding contradictions and ensuring the information is biomedically right. Generate a highly accurate "comprehensive observation report.

## Bad example:It combines Qwenvl's incorrect "cytoplasmic" description with a correct "prominent nuclear staining" description later,  resulting in a confusing and anatomically impossible description for a single stain.

# Input Observations:

[Model: Fleming]:
{desc_fleming}

[Model: Hulu]:
{desc_hulu}

[Model: Lingshu]:
{desc_lingshu}

[Model: QwenVL]:
{desc_qwenvl}

# Key Requirements:
1. Voting & Merging Strategy: 
    - For overlapping features mentioned by multiple models, use **majority voting** to establish the **corroborated facts**.
    - For distinct/unique details mentioned by only one model, **naturally merge** them into the description to enrich detail, provided they DO NOT contradict the **corroborated facts** or biomedical logic.

2. Pure Observation: Describe ONLY the visible morphological features ( e.g. cells, staining, structures   ). Do not include any reasoning, such as "consistent with...", "indicates...", "seems to...", "suggests some expression...", "may represent...".  Focus on the visual features of the image itself; do not draw conclusions or inferences based on interpretations or deductions from the image.
3. Integration: Output a single, coherent paragraph merging the corroborated facts and valid unique details naturally.

# Output
(Output ONLY the Integration description paragraph.)
"""

ENHANCED_CAPTION_PROMPT_TEMPLATE = """
You are an expert biologist and biomedical researcher. You will be given [Context], [Background], [Keywords], and a set of initial [Image_captions].

# Your goal is to generate a "Context_Enhanced_Captions" object. You must process the data in two distinct steps for each image: Verification ([Observation]) and Analysis ([Interpretation]).

## Verification ([Observation]): Rigorously validate the [Image_captions] to correct only factual errors based on [Context] while strictly preserving all non-conflicting visual details, ensuring the output remains a purely descriptive report devoid of any explanatory logic.

## Analysis ([Interpretation]): Synthesize the verified visual observations with [Context], [Background], and [Keywords] to explain the deep biological principles, mechanisms, or pathologies underlying the visual data.

# Input Data

[Background]:
{distilled_background}

[Keywords]:
{keywords}

[Context]:
{context}

[Image_captions]:
{vl_captions_json}

# Task Guidelines & Logic

# 1. Alignment Strategy
The [Context] text contains `[Image]` tokens (e.g., [Image 1], [Image 2]). These tokens mark the exact location where the image is discussed.
You must use the text immediately surrounding these `[Image]` tokens to verify the identity and features of the corresponding image in [Image_captions].

# 2. Field: "observations" (Strict Visual Verification)
Goal: Correct the [Image_captions] ONLY if they are factually wrong based on the [Context], while preserving correct visual details.
## Minimal Modification Rule: Do not rewrite the caption if it is consistent with the text. Only edit specific words or phrases that contradict the [Context].
### Correction Protocol:
    * If [Image_captions] says "blue stain" but [Context] specifies "red stain", change it to "red stain".
    * If [Image_captions] mentions a visual detail (e.g., "irregular shape") that is NOT mentioned in [Context], PRESERVE IT. Do not delete valid visual details just because the text doesn't mention them.
### Anti-Hallucination Rule**: Do NOT add biological reasoning, causal relationships, or background knowledge into this field. Keep it purely descriptive (shapes, colors, positions ).

# 3. Field: "interpretations" (Biological Reasoning)
Goal: Explain the biological significance of the verified [Observation] using [Background] and [Context].
* Use this field to bridge the gap between "what we see" ([Observation]) and "what it means" (Context).
* Explain the function, process, or pathology visible in the image.
* Synthesize information from the [Background] to provide depth.

# 4. Summary Generation
## [Observation] summary: If there is only one image, provide a concise visual overview of that specific image. If there are multiple images, synthesize the common visual themes across all panels. MUST remain purely descriptive (no reasoning).
## [Interpretation] summary: If there is only one image, provide a final biological conclusion or diagnosis based on the analysis of the single image. If there are multiple images,   Provide a joint analysis of the overall biological conclusion derived from the combination of these images.

# Output Format
Provide the final answer as a JSON object with a single root key "Context_Enhanced_Captions".
The output must strictly separate visual descriptions from analytical insights.
Ensure the output is a valid JSON list of strings within the structure, corresponding one-to-one with the original captions.

```json
{{
  "Context_Enhanced_Captions": {{
    "observations": {{
      "Image 1": "...",
      "Image 2": "...",
      ...
      "summary": "..."
    }},
    "interpretations": {{
      "Image 1": "...",
      "Image 2": "...",
      ...
      "summary": "..."
    }}
  }}
}}

# Key Requirements
1. JSON Validity: The output must be directly parseable by json.loads.

2. Separation of Concerns:
"[Observation]" = Pure Vision + Contextual Correction (No "because...", No "indicating that...").
"[Interpretation]" = Vision + Contextual Logic (Explain the "Why").
3. Context Fidelity: Do not hallucinate details not present in the image or the text.
4. Count Match: The number of keys in the dictionary must match the number of input images.
5. Keyword Integration: Utilize the [Keywords] to anchor your terminology in "[Interpretation]", ensuring the specific modality, staining technique, or pathological classification is accurately reflected.

Generate [Context-Enhanced Captions]: 
"""

VISUAL_ELEMENT_QA_PROMPT_TEMPLATE = """
You are an expert in biomedical image analysis. Your task is to generate a vision-centered [Question]-[Answer] pairs based ONLY on the provided [Observation].Extract a list of all unique biomedical entities (e.g., cell types, staining, anatomical structures) mentioned in the [Observation].

# Input Data
[Observation] refers to objective visual descriptions of each images, and the summary consolidates the visual findings to provide a holistic overview,inter image relationship across the images
{Observation}

# Output

Generate a [Question]-[Answer] pair that asks for the specific biomedical visual features mentioned in the [Observation]. If there is only one image available, then only use that single image for the question.If there are multiple images, selecting several (but not necessarily all) closely related images can generate reasonable questions.
The goal is to verify if the model can "see" the low-level details before performing high-level reasoning.

## Key Requirements (MUST FOLLOW):
1. Strictly Visual: The [Question] MUST focus ONLY on visual attributes. Aim for high diversity in question suitable for multi-image biomedical analysis. 
IF [Observation] contains Only One Image**: You MUST generate a Descriptive question specific to that image (e.g., "Describe the staining intensity of the cytoplasm in Image 1."). Do NOT hallucinate other images.

IF [Observation] contains multiple images, you can generate questions about the relationships between these images.

Examples include:
Comparative Morphology: 'Compare the nuclear irregularity observed in different images.'
Feature Characterization: 'Describe the texture and staining intensity of the cytoplasm in...'
Structural Architecture: 'How does the arrangement of inflammatory cells differ between different images?'

2. No Interpretation: The [Question] and [Answer] MUST NOT contain diagnostic conclusions, biological significance, or "Why" reasoning. Do not use words like "suggests", "indicates", or "diagnosis".
3. Image Reference in [Question]:
    - **IF [Observation] contains Multiple Images**:  The [Question] string MUST include at least 2 explicit image references (e.g., [Image 1, Image2, Image3, ...]).
    - **IF [Observation] contains Only One Image**: You MUST generate a  question specific to that image. Do NOT hallucinate other images.The [Question] string MUST include  explicit image references (e.g., Image 1).
4. Fact-Based: The [Answer] must be rely on the [Observation] text.
5. Atomic [Question]: the [Question] string must be a single query. It MUST NOT be a compound question or contain any sub-questions.

## OUTPUT FORMAT AND CONSTRAINTS (MUST FOLLOW):
Return a valid JSON **List** of objects:
```json
[
  {{
    "qa_pairs" : {{
      "qa1": {{
        "question": "...",
        "answer": "...",
      }},
      "qa2": {{
        "question": "...",
        "answer": "...",
      }},
      ...
    }}
    "image_indices": [...],
    "biomedical_entities": ["entity1", "entity2", "..."]
  }}
]

Task
Generate exactly mltiple visual description QA pair based on the [Observation]. 
2. Extract a list of all unique biomedical entities (e.g., cell types, staining, anatomical structures) mentioned in the [Observation] and put them in "biomedical_entities".
[Your Output]: """

LOGIC_CHAIN_PROMPT_TEMPLATE = """
You are a rigorous biomedical expert. You need to construct logical reasoning chains based on the provided Original Text and Visual Evidence.

Original Text:
{context}

Visual Evidence:
{observation}

Please integrate the Context and Visual Evidence to form detailed logical reasoning chains that lead to conclusions.
Requirements:
- Each independent research in the Context should correspond to a separate logical reasoning chain.
- Each logical reasoning chain should follow the logic:
  Research Context -> Experiments -> Conclusion
- Each Experiment should follow the logic:
  Experimental Setting -> Experiment Goal -> Visual Phenomenon -> Interpretation -> Sub-Conclusion
    - Experimental Setting: Describe the experimental setup, including materials, methods, and conditions.
    - Experiment Goal: Purpose of the experiment.
    - Visual Phenomenon: Specific **visual** observations from the experiment, not interpretations.
    - Interpretation: Scientific explanation of the visual phenomenon.
    - Sub-Conclusion: Conclusion drawn from the interpretation, related to the final conclusion.
- If a visual phenomenon of experiment is mentioned in the Visual Evidence, mark which image it corresponds to in the format [Image X].
- Some experiments may not have any visual phenomenon in the Visual Evidence. There are two cases:
    - The Context provides the visual phenomenon directly. In this case, provide the visual phenomenon and mark it as [Context].
    - The visual phenomenon is missing. In this case, provide [Missing] as the visual phenomenon.
- Avoid precise numerical measurement data unless exactly the same numbers are present in Experimental Setting.
- The process of achieving the conclusion should include all necessary intermediate sub-conclusions and corresponding experiments.
- Each logical reasoning chain should end with a clear conclusion.
- If a certain experiment has no contribution to the final conclusion, it should be omitted from the whole logical reasoning chain.
Output Format:
Provide the logical reasoning chains in JSON format as a list of objects with the following structure:
```json
[
  {{
    "research_context": "Description of the research context.",
    "experiments": [
      {{
        "experimental_setting": "Description of the experimental setting.",
        "experiment_goal": "Description of the experiment goal.",
        "visual_phenomenon": "Visual phenomenon details with [Image X] or [Context] or [Missing].",
        "interpretation": "Interpretation of the visual phenomenon.",
        "sub_conclusion": "Conclusion drawn from the interpretation, related to the final conclusion."
      }},
      ...
    ],
    "reasoning": {{
      "intermediate_inferences": [
        {{
          "sub_conclusion": "Description of the intermediate inference.",
          "based_on_experiments": [Indices of experiments contributing to this inference]
        }},
        ...
      ],
      "content": "Detailed reasoning process leading to the conclusion.",
      "conclusion": "Final conclusion derived from the reasoning."
    }}
  }}
]
```
"""

OPEN_ENDED_QA_GENERATION_PROMPT_TEMPLATE = """
You are a biomedical expert.
You are given a logic chain, and you need to generate a exam question to test students' comprehension of the logic chain.

Logic Chain:
{logic_chain}

Some extra information that may help you:
Visual Evidence:
{visual_evidence}

Original Text:
{original_text}

The questions is expected to be hard, requiring both accurate observations and deep understanding of the logic chain. This includes the following aspects:
- For the answer:
    - The question should be open-ended.
    - The answer should contain the whole logical reasoning chain above.

- For the information provided in the question:
    - The Research Context should be provided.
    - The Setting of each Experiment should be provided.
    - The Goal of each Experiment should NEVER be provided.
    - For Visual Phenomenon of each Experiment:
        - If at least one visual phenomenon of the experiment is mentioned in the Visual Evidence, do NOT provide the Visual Phenomenon or Result. I.e. sentence like "[Image X] shows ..." should NEVER appear in the question.
          - Only one EXEMPT: If Visual Phenomenon contains Scale Bars, mention the scale ratio in the question.
        - If all visual phenomena of the experiment are provided in Context, provide the Visual Phenomenon. Do NOT provide the Result.
        - If the visual phenomenon is marked as Missing, provide the experiment result instead.
    - The direct result (NOT further Interpretation or Sub-conclusion) of each experiment should be provided only if the Visual Phenomenon is marked as Missing.
    - The Interpretation and Sub-Conclusion of each Experiment should NEVER be provided.
    - Intermediate Inferences should NEVER be provided.
    - Reasoning from Intermediate Inferences to Conclusion should NEVER be provided.
    - The Conclusion should NEVER be provided.

- For how to ask the question:
    - The question should not easily guide students to the answer. That is:
        - The question should not give any clues about how to reason to the answer.
        - The question should not give away intermediate steps or conclusions.

For example:
The logic chain is:

Research Context RC
Experiment E1:
  Setting: S1
  Visual Phenomenon: P1 [Image X]
  Interpretation: I1
  Sub-Conclusion: SC1
Experiment E2:
  Setting: S2
  Visual Phenomenon: P2 [Context]
  Interpretation: I2
  Sub-Conclusion: SC2
Experiment E3:
  Setting: S3
  Visual Phenomenon: [Missing]
  Interpretation: I3
  Sub-Conclusion: SC3
Final Reasoning:
  Based on SC1, SC2 and SC3, we conclude Conclusion C.

Do NOT ask:
- [BAD CASE] "How is conclusion C derived?" (gives away the conclusion)
- [BAD CASE] "Research RC, conducted experiment E1, setting S1, observed P1, ..." (gives away visible phenomenon in the images)
- [BAD CASE] "Research RC, ..., conducted experiment E2, setting S2, result R2, ..." (gives away experiment result where phenomenon is provided in Context)
- [BAD CASE] "Research RC, ..., conducted experiment E3, interpretation I3, ..." (gives away interpretation)
- [BAD CASE] "Research RC, conducted experiment E1, (Did not provide S1), ..." (misses the setup of an experiment)
- [BAD CASE] "Research RC, ..., conducted experiment E2, setting S2, (Did not provide P2), ..." (misses the visual phenomenon that is not provided in the images but provided in Context)
- [BAD CASE] "Research RC, ..., conducted experiment E3, setting S3, (Neither phenomenon or direct result), ..." (misses the direct result of an experiment where the visual phenomenon is marked as Missing)
- [BAD CASE] "Based on SC1, SC2 and SC3, what is the conclusion?" (gives away intermediate inferences)
Ask instead:
[GOOD CASE] "Research RC, conducted experiment E1, setting S1; conducted experiment E2, setting S2, observed P2; conducted experiment E3, setting S3, got result R3. What can be concluded from these experiments?"
Note that you do not need to directly ask "Please give a detailed reasoning process". A clever student should know to provide the reasoning process to reach the conclusion.


Format your output as a JSON object with three fields: "question" and "answer", where "question" contains the generated question, "answer" and "explanation", where "explanation" explains how you generated this question-answer pair according to the requirements above.
```json
{{
  "explanation": "{{your explanation here}}",
  "question": "{{our question here}}",
  "answer": "{{your answer here}}"
}}
```
"""

LOGIC_CHAIN_QC_1_TEMPLATE = \
"""
You are an expert in biomedical reasoning and logic evaluation.
Your task is to evaluate the integrity and coherence of a logic chain.
The input is a structured list of strings representing the progression from experimental facts to intermediate inferences, and finally to a conclusion.

# Input Data

[Logic Chain] (The reasoning path to evaluate):
{flattened_logic_chain}

# Evaluation Criteria (1-5 Scale)

1. Evidence Support Strength
   Assess if the intermediate inferences provide sufficient and accurate support for the final reasoning content.
   - Score 1 (Critical Fail): Contradictory or Unsupported. The final content makes claims that contradict the intermediate inferences or relies on evidence not present in the chain.
   - Score 3 (Borderline): Weak or Partial Support. The final content is somewhat related but contains major leaps in logic or includes details not fully backed by the intermediate steps.
   - Score 5 (Pass): Strong Support. The final content is a robust and accurate synthesis strictly derived from the provided intermediate inferences.

2. Logical Flow and Coherence
   Assess if the transition from Intermediate Inferences to the Final Conclusion is logically sound and seamless.
   - Score 1 (Critical Fail): Fragmented or Disjointed. The logic jumps randomly; the connection between the inference layer and the conclusion layer is broken or nonsensical.
   - Score 3 (Borderline): Rough or Repetitive. The flow is understandable but clunky, redundant, or requires the reader to guess the connection between steps.
   - Score 5 (Pass): Seamless and Coherent. The reasoning flows naturally like a scientific argument; the conclusion feels like the inevitable result of the preceding steps.

# Output Format (Strict JSON)

You must return the result strictly in the following format:

<scores>
{{
  "Evidence Support Strength": A,
  "Logical Flow and Coherence": B
}}
</scores>

<explanation>
[Provide a brief explanation for your scoring. explicitly stating if there are logical gaps, contradictions, or if the chain is solid.]
</explanation>

(Where A, B are integer scores from 1 to 5)
"""

LOGIC_CHAIN_QC_2_TEMPLATE = \
"""
You are an expert in biomedical text verification and fact-checking.
Your task is to verify if the [Visual Phenomena] described in the logic chain are supported by the provided Source Data ([Observation] and [Context]).

# Input Data

[Observation] (Objective visual descriptions of the images):
{Observation}

[Context] (Background containing [Image] tags):
{Context}

[Visual Phenomena] (The descriptions extracted from the logic chain to be verified):
{VisualPhenomena}

# Evaluation Criteria (1-5 Scale)

1. Source Grounding & Verification
   Assess if every visual phenomenon listed in the Target is explicitly mentioned or clearly visible in the [Context] or [Observation].
   - Score 1 (Critical Fail): Hallucination. The target describes features that are completely absent from both the Observation and Context, or contradicts them.
   - Score 3 (Borderline): Partial Match. Some descriptions are supported, but others are missing source evidence, or the target adds significant details not found in the source.
   - Score 5 (Pass): Fully Grounded. Every statement in the [Visual Phenomena] is directly supported by evidence found in the Source Observation or Source Context (textual descriptions of visual outcomes).

# Output Format (Strict JSON)

You must return the result strictly in the following format:

<scores>
{{
  "Source Grounding & Verification": A
}}
</scores>

<explanation>
[Provide a brief explanation. If there is a hallucination or missing reference, explicitly quote the unsupported part.]
</explanation>

(Where A is an integer score from 1 to 5)
"""

LOGIC_CHAIN_QC_3_TEMPLATE = \
"""
You are an expert in evaluating question-answering logic.
Your task is to verify if the provided Conclusion effectively answers or corresponds to the specific Question asked.

# Input Data

Question:
{Question}

Observation (Visual Evidence containing scale info):
{Observation}

Logic Chain:
{LogicChain}

Conclusion (Derived from Logic Chain):
{Conclusion}

# Evaluation Criteria (1-5 Scale)

1. Question-Conclusion Alignment
   Assess if the Conclusion directly addresses the core inquiry of the Question.
   - Score 1 (Fail): The conclusion is irrelevant, unrelated, or contradicts the premise of the question. It does not provide an answer.
   - Score 3 (Passable): The conclusion is related and provides a partial answer, but may be slightly tangential or misses the specific format requested.
   - Score 5 (Pass): The conclusion provides a clear, logical, and direct answer to the question. It functions effectively as the final output.

2. Scale/Legend Consistency Check
Check if the problem statement lacks a scale/legend, but the observation results, reasoning content, and conclusion clearly include scale numbers or scale information.
- Score 1 point (Serious Failure): The problem statement lacks a scale/legend, but the observation results contain explicit scale numbers (e.g., "50 nm," "scale"), and the reasoning content utilizes this scale information from the observation.
- Score 5 points (Pass): The problem statement and observation results are consistent; either both include a scale/legend, or neither includes scale-related information. If the problem statement includes scale-related information, but the conclusion and reasoning content do not use it, it is not considered an error.

3. Reasoning Validity 
   Assess if the Logic Chain steps contains excessive speculation or hallucinations not supported by the Observation.
   - Score 1 (Critical Fail): Given ONLY Research Context, Experimental Settings, and Visual Phenomenon, the "inference", "sub_conclusion", "content", "conclusion" parts contains details impossible to know.
   - Score 5 (Pass): Given ONLY Research Context, Experimental Settings, and Visual Phenomenon, the "inference", "sub_conclusion", "content", "conclusion" parts are all supported without any hallucination.

# Output Format (Strict JSON)

You must return the result strictly in the following format:

<scores>
{{
  "Question-Conclusion Alignment": A,
  "Scale/Legend Consistency Check" : B,
  "Reasoning Validity" : C
}}
</scores>

<explanation>
[Briefly explain why the conclusion satisfies or fails to answer the question.]
</explanation>


(Where A, B, C is an integer score from 1 to 5)
"""

# ================= UTILITY FUNCTIONS =================


def openai_pack_content(prompt, images):
    image_list = images or []
    content = [{
        "type": "image_url",
        "image_url": {
            "url": img_url,
            "detail": "auto"
        }
    } for img_url in image_list] + [{
        "type": "text",
        "text": prompt
    }]
    return content


def process_qa_output(output_str):
    output_str = output_str.strip()
    if output_str.startswith("```json") and output_str.endswith("```"):
        output_str = output_str[len("```json"):-len("```")].strip()
    try:
        qa_list = json.loads(output_str)
        return qa_list
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None


async def get_response_async(prev_messages,
                             next_content,
                             model,
                             client,
                             tools=None,
                             max_retries=3):
    if isinstance(next_content, str):
        user_content = next_content
    else:
        user_content = next_content

    messages = prev_messages + [{"role": "user", "content": user_content}]
    MAX_TOKENS_LIMIT = 4096

    for attempt in range(max_retries):
        try:
            answer_content = ""
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS_LIMIT,
                temperature=0.2)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    answer_content += chunk.choices[0].delta.content

            return {"content": answer_content}

        except (APIConnectionError, InternalServerError) as e:
            print(
                f"--- [Retryable Error] (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1: raise e
            await asyncio.sleep(5)
        except Exception as e:
            print(f"--- [Fatal Error]: {e}")
            raise e


def get_image_base64_from_source(img_info):
    if img_info.get("image_base64"):
        return img_info["image_base64"]
    local_path = img_info.get("local_path")
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except:
            return None
    return None


def split_caption_data(item):
    captions_list = item.get("context_enhanced_captions", [])
    summary_data = item.get("context_enhanced_summary", {})
    obs_parts = []
    int_parts = []
    for entry in captions_list:
        idx = entry.get("image_index")
        f_id = entry.get("fig_id", "")
        s_label = entry.get("subfig_label", "")
        id_prefix = f":{f_id} {s_label} :" if (f_id or s_label) else ""

        obs = entry.get("observation", "")
        if obs and obs != "Not found":
            obs_parts.append(f"{id_prefix} [Image {idx}]: {obs}")

        interp = entry.get("interpretation", "")
        if interp and interp != "Not found":
            int_parts.append(f"{id_prefix} [Image {idx}]: {interp}")

    if summary_data:
        if summary_data.get("observation_summary"):
            obs_parts.append(
                f"[Observation Summary]: {summary_data['observation_summary']}"
            )
        if summary_data.get("interpretation_summary"):
            int_parts.append(
                f"[Interpretation Summary]: {summary_data['interpretation_summary']}"
            )

    return "\n".join(obs_parts), "\n".join(int_parts)


def extract_visual_info(visual_qa_data):
    if not visual_qa_data: return "{}"
    filtered_data = {}
    raw_pairs = visual_qa_data.get("qa_pairs", {})
    processed_pairs = {}
    for i, (original_key, val) in enumerate(raw_pairs.items(), start=1):
        processed_pairs[f"{i}"] = val.get("answer", "")
    filtered_data["visual_facts"] = processed_pairs
    return json.dumps(filtered_data, ensure_ascii=False, indent=2)


def extract_observation_text(item):
    captions_list = item.get("context_enhanced_captions", [])
    summary_data = item.get("context_enhanced_summary", {})
    obs_summary = summary_data.get("observation_summary", "")
    combined_obs_parts = []
    for entry in captions_list:
        idx = entry.get("image_index")
        obs_text = entry.get("observation", "")
        if obs_text and obs_text != "Not found":
            combined_obs_parts.append(f"[Image {idx}]: {obs_text}")
    if obs_summary:
        combined_obs_parts.append(f"[Summary]: {obs_summary}")
    return "\n".join(combined_obs_parts) if combined_obs_parts else None


# ================= STEP FUNCTIONS =================


# --- Step 0: Filter ---
async def check_biomedical_async(sample):
    try:
        back_info = sample.get("back_info", "")
        if not back_info:
            text_list = sample.get("text_list", [])
            back_info = " ".join([t for t in text_list if isinstance(t, str)])

        context_for_judge = back_info
        prompt = BIOMED_CHECK_PROMPT.format(context=context_for_judge)
        response = await get_response_async([], prompt, TEXT_MODEL_NAME,
                                            local_text_client)
        content = response["content"].strip()

        if content.startswith("```json"): content = content[7:].strip()
        if content.endswith("```"): content = content[:-3].strip()

        is_biomedical = False
        try:
            res_json = json.loads(content)
            is_biomedical = res_json.get("is_biomedical", False)
        except:
            if "true" in content.lower(): is_biomedical = True

        if is_biomedical:
            raw_image_info = sample.get("image_info", [])
            formatted_captions = []
            for i, img in enumerate(raw_image_info):
                formatted_captions.append({
                    "image_index":
                    i + 1,
                    "caption":
                    img.get("caption", ""),
                    "fig_id":
                    img.get("fig_id", ""),
                    "subfig_label":
                    img.get("subfig_label", "")
                })
            lightweight_sample = {
                "original_sample_index": sample.get("original_sample_index"),
                "text_list": sample.get("text_list", []),
                "back_info": sample.get("back_info", ""),
                "image_captions": formatted_captions
            }
            return {"status": "valid", "data": lightweight_sample}
        else:
            return {
                "status": "filtered",
                "original_sample_index": sample.get("original_sample_index")
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "original_sample_index": sample.get("original_sample_index")
        }


# --- Step 1: Keywords ---
async def extract_keywords_from_filtered_async(sample):
    orig_idx = sample.get("original_sample_index", "N/A")
    try:
        text_list = sample.get("text_list", [])
        image_captions = sample.get("image_captions", [])
        modified_text_list = []
        image_insert_counter = 0
        num_available_images = len(image_captions)

        for text in text_list:
            has_images_left = (image_insert_counter < num_available_images)
            if text == "" and has_images_left:
                modified_text_list.append(
                    f" [Image {image_insert_counter + 1}]")
                image_insert_counter += 1
            elif isinstance(text,
                            str) and text.startswith(")") and has_images_left:
                modified_text_list.append(
                    f" [Image {image_insert_counter + 1}]{text}")
                image_insert_counter += 1
            else:
                modified_text_list.append(str(text))

        context = "".join(modified_text_list)
        formatted_captions_for_llm = [
            f"Image {img['image_index']}: {img['caption']}"
            for img in image_captions
        ]

        kw_prompt = KEYWORD_Category_PROMPT_TEMPLATE.format(
            context=context,
            image_caption=json.dumps(formatted_captions_for_llm,
                                     ensure_ascii=False,
                                     indent=2))
        response = await get_response_async([], kw_prompt, TEXT_MODEL_NAME,
                                            local_text_client)

        return {
            "status": "success",
            "original_sample_index": orig_idx,
            "context": context,
            "image_captions": image_captions,
            "extracted_keywords": response["content"].strip(),
            "back_info": sample.get("back_info", "")
        }
    except Exception as e:
        return {
            "status": "failed",
            "original_sample_index": orig_idx,
            "error": str(e)
        }


# --- Step 2: Distillation ---
async def process_single_sample_background(result_item):
    oid = result_item.get("original_sample_index")
    back_info = result_item.get("back_info")
    if not back_info: return oid, {"status": "skipped"}
    if len(back_info.split()) < 200:
        return oid, {"status": "success", "distilled_background": back_info}

    prompt = BACKGROUND_DISTILLATION_PROMPT_TEMPLATE.format(
        back_info=back_info)
    try:
        response = await get_response_async([], prompt, TEXT_MODEL_NAME,
                                            local_text_client)
        return oid, {
            "status": "success",
            "distilled_background": response['content'].strip()
        }
    except Exception as e:
        return oid, {"status": "error", "error_message": str(e)}


# --- Step 2b: VLM Enhancement ---
async def process_sample_vlm(lightweight_item, source_map, semaphore):
    orig_id = lightweight_item.get("original_sample_index")
    raw_sample = source_map.get(orig_id) or source_map.get(str(orig_id))
    if not raw_sample: return orig_id, []

    image_info_list = raw_sample.get("image_info", [])
    enhanced_results = []

    for img_info in image_info_list:
        base64_str = get_image_base64_from_source(img_info)
        if not base64_str:
            enhanced_results.append("ERROR: Image data missing")
            continue

        prompt = VLM_PROMPT_TEMPLATE.format(
            caption=img_info.get("caption", ""))
        async with semaphore:
            try:
                response = await local_vl_client.chat.completions.create(
                    model=VL_MODEL_NAME,
                    messages=[{
                        "role":
                        "user",
                        "content": [{
                            "type": "text",
                            "text": prompt
                        }, {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_str}"
                            }
                        }]
                    }],
                    max_tokens=512,
                    temperature=0.2)
                enhanced_results.append(response.choices[0].message.content)
            except Exception as e:
                enhanced_results.append(f"ERROR: {e}")

    structured_captions = [{
        "image_index":
        i + 1,
        "description":
        str(res).replace("[Enhanced Captions]:", "").strip()
    } for i, res in enumerate(enhanced_results)]
    return orig_id, structured_captions


# --- Step 3.5: Consensus ---
async def process_sample_consensus(item):
    # This is a simplified version, assuming only the results from the current VLM are used.
    # If there are other model files, they need to be loaded and merged into `desc_map` here.
    desc_map = {
        "qwenvl": "",  # Placeholder
    }


    vlm_caps = item.get("model-enhanced captions", [])
    vlm_map = {x["image_index"]: x["description"] for x in vlm_caps}

    output_entry = item.copy()
    output_entry["consensus_image_descriptions"] = []

    image_info_list = item.get("image_captions", [])
    for img_info in image_info_list:
        idx = img_info.get("image_index")
        desc_map["qwenvl"] = vlm_map.get(idx, "")

        # If only a single model is used, directly use the single model's result, or generate a consensus based on the prompt requirements.
        # Here, for demonstration purposes, we call the Consensus Prompt, even though there is only one input.
        prompt = CONSENSUS_PROMPT_TEMPLATE.format(
            desc_fleming="N/A",
            desc_hulu="N/A",
            desc_lingshu="N/A",
            desc_qwenvl=desc_map["qwenvl"])
        try:
            res = await get_response_async([], prompt, TEXT_MODEL_NAME,
                                           local_text_client)
            consensus_text = res['content'].strip()
        except:
            consensus_text = desc_map["qwenvl"]  # Fallback

        output_entry["consensus_image_descriptions"].append({
            "image_index":
            idx,
            "description":
            consensus_text
        })
    return output_entry


# --- Step 3: Context Enhanced ---
async def process_enhanced_caption_gen(item):
    captions_json_str = json.dumps(item.get("consensus_image_descriptions",
                                            []),
                                   indent=2,
                                   ensure_ascii=False)
    prompt = ENHANCED_CAPTION_PROMPT_TEMPLATE.format(
        distilled_background=item.get("distilled_background", ""),
        keywords=item.get("extracted_keywords", ""),
        context=item.get("context", ""),
        vl_captions_json=captions_json_str)
    try:
        response = await get_response_async([], prompt, TEXT_MODEL_NAME,
                                            local_text_client)
        content = response['content'].strip()
        if "```" in content:
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match: content = match.group(1).strip()

        data = json.loads(content)
        core_data = data.get("Context_Enhanced_Captions", data)

        obs_dict = core_data.get("observations", {})
        interp_dict = core_data.get("interpretations", {})

        structured_captions = []
        original_caps = item.get("image_captions", [])

        for i in range(len(original_caps)):
            idx = i + 1
            key = f"Image {idx}"
            structured_captions.append({
                "image_index":
                idx,
                "fig_id":
                original_caps[i].get("fig_id"),
                "subfig_label":
                original_caps[i].get("subfig_label"),
                "observation":
                obs_dict.get(key, "Not found"),
                "interpretation":
                interp_dict.get(key, "Not found")
            })

        item["context_enhanced_captions"] = structured_captions
        item["context_enhanced_summary"] = {
            "observation_summary": obs_dict.get("summary", ""),
            "interpretation_summary": interp_dict.get("summary", "")
        }
        return item
    except Exception as e:
        print(f"Enhance Error: {e}")
        return None


# --- Step 4: Visual QA ---
async def run_visual_qa_task(item):
    obs_str = extract_observation_text(item)
    if not obs_str: return None

    prompt = VISUAL_ELEMENT_QA_PROMPT_TEMPLATE.format(Observation=obs_str)
    try:
        result = await get_response_async([], prompt, TEXT_MODEL_NAME,
                                          local_text_client)
        content = result["content"].strip()
        if "```" in content:
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match: content = match.group(1).strip()

        qa_list = json.loads(content)
        if not qa_list: return None

        result_data = item.copy()
        result_data["visual_qa"] = qa_list[0]
        return result_data
    except Exception as e:
        print(f"Visual QA Error: {e}")
        return None


# --- Step 5: Logic Chain ---
async def run_logic_chain_task(item):
    visual_fact = extract_visual_info(item.get("visual_qa", {}))
    obs_str, _ = split_caption_data(item)
    context_str = item.get("context", "")

    prompt = LOGIC_CHAIN_PROMPT_TEMPLATE.format(observation=obs_str,
                                                context=context_str)
    try:
        content_packed = openai_pack_content(prompt, None)
        result = await get_response_async([], content_packed, TEXT_MODEL_NAME,
                                          local_text_client)
        logic_chain_json = process_qa_output(result["content"])
        if not logic_chain_json: raise Exception("Empty logic chain")

        result_data = item.copy()
        result_data["logic_chain"] = logic_chain_json
        return result_data
    except Exception as e:
        print(f"Logic Chain Error: {e}")
        return None


# --- Step 6: Logic QA ---
async def run_logic_based_qa_task(item):
    logic_chain = item.get("logic_chain", {})
    obs_str, _ = split_caption_data(item)
    context_str = item.get("context", "")

    prompt = OPEN_ENDED_QA_GENERATION_PROMPT_TEMPLATE.format(
        logic_chain=json.dumps(logic_chain[0], ensure_ascii=False, indent=2),
        visual_evidence=obs_str,
        original_text=context_str)
    try:
        content_packed = openai_pack_content(prompt, None)
        result = await get_response_async([], content_packed, TEXT_MODEL_NAME,
                                          local_text_client)
        qa_pair = process_qa_output(result["content"])

        result_data = {
            "original_sample_index": item.get("original_sample_index"),
            "basic_qa": qa_pair,
            "input_observation": obs_str,
            "input_context": context_str,
            "input_logic_chain": logic_chain
        }
        return result_data
    except Exception as e:
        print(f"Logic QA Error: {e}")
        return None


# ================= MAIN EXECUTION =================


async def main():
    # The `data_final` file corresponds to the data in `qa_generation_quickly.json`, and contains only filtered biomedical data. If you want to see the results of the first filtering step, you can use `source_data.json`, which includes non-biomedical data.
    INPUT_SOURCE_FILE = 'Data\qa_generation_quickly.json'

    FILE_STEP0 = "step0_filtered.json"
    FILE_STEP1 = "step1_keywords.json"
    FILE_STEP2 = "step2_distilled.json"
    FILE_STEP2B = "step2b_vlm.json"
    FILE_STEP3_5 = "step3_5_consensus.json"
    FILE_STEP3 = "step3_enhanced.json"
    FILE_STEP4 = "step4_visual_qa.json"
    FILE_STEP5 = "step5_logic_chain.json"
    FINAL_OUTPUT = "final_qa_dataset.json"


    if not os.path.exists(INPUT_SOURCE_FILE):
        print(f"Error: {INPUT_SOURCE_FILE} not found.")
        return
    with open(INPUT_SOURCE_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} items.")

    # 2. Step 0: Filter
    print("\n--- Running Step 0: Filtering ---")
    tasks = [check_biomedical_async(s) for s in raw_data]
    res0 = await tqdm_asyncio.gather(*tasks)
    step0_data = [r["data"] for r in res0 if r["status"] == "valid"]
    with open(FILE_STEP0, 'w', encoding='utf-8') as f:
        json.dump(step0_data, f, indent=2, ensure_ascii=False)

    # 3. Step 1: Keywords
    print(f"\n--- Running Step 1: Keywords ({len(step0_data)} items) ---")
    tasks = [extract_keywords_from_filtered_async(s) for s in step0_data]
    res1 = await tqdm_asyncio.gather(*tasks)
    step1_data = [r for r in res1 if r["status"] == "success"]
    with open(FILE_STEP1, 'w', encoding='utf-8') as f:
        json.dump(step1_data, f, indent=2, ensure_ascii=False)

    # 4. Step 2: Background Distillation
    print("\n--- Running Step 2: Distillation ---")
    tasks = [process_single_sample_background(s) for s in step1_data]
    res2 = await tqdm_asyncio.gather(*tasks)
    res2_map = {oid: r.get("distilled_background", "") for oid, r in res2}
    for item in step1_data:
        item["distilled_background"] = res2_map.get(
            item["original_sample_index"], "")
    with open(FILE_STEP2, 'w', encoding='utf-8') as f:
        json.dump(step1_data, f, indent=2, ensure_ascii=False)

    # 5. Step 2b: VLM (Images)
    print("\n--- Running Step 2b: VLM Enhancement ---")
    source_map = {
        item["original_sample_index"]: item
        for item in raw_data
    }  # Map back to raw for images
    sem_vlm = asyncio.Semaphore(10)
    tasks = [process_sample_vlm(s, source_map, sem_vlm) for s in step1_data]
    res2b = await tqdm_asyncio.gather(*tasks)
    res2b_map = {oid: caps for oid, caps in res2b}
    for item in step1_data:
        item["model-enhanced captions"] = res2b_map.get(
            item["original_sample_index"], [])
    with open(FILE_STEP2B, 'w', encoding='utf-8') as f:
        json.dump(step1_data, f, indent=2, ensure_ascii=False)

    # 6. Step 3.5: Consensus
    print("\n--- Running Step 3.5: Consensus ---")
    tasks = [process_sample_consensus(s) for s in step1_data]
    step3_5_data = await tqdm_asyncio.gather(*tasks)
    with open(FILE_STEP3_5, 'w', encoding='utf-8') as f:
        json.dump(step3_5_data, f, indent=2, ensure_ascii=False)

    # 7. Step 3: Context Enhanced
    print("\n--- Running Step 3: Enhancement ---")
    tasks = [process_enhanced_caption_gen(s) for s in step3_5_data]
    step3_data = [
        r for r in await tqdm_asyncio.gather(*tasks) if r is not None
    ]
    with open(FILE_STEP3, 'w', encoding='utf-8') as f:
        json.dump(step3_data, f, indent=2, ensure_ascii=False)

    # 8. Step 4: Visual QA
    print("\n--- Running Step 4: Visual QA ---")
    tasks = [run_visual_qa_task(s) for s in step3_data]
    step4_data = [
        r for r in await tqdm_asyncio.gather(*tasks) if r is not None
    ]
    with open(FILE_STEP4, 'w', encoding='utf-8') as f:
        json.dump(step4_data, f, indent=2, ensure_ascii=False)

    # 9. Step 5: Logic Chain
    print("\n--- Running Step 5: Logic Chain ---")
    tasks = [run_logic_chain_task(s) for s in step4_data]
    step5_data = [
        r for r in await tqdm_asyncio.gather(*tasks) if r is not None
    ]
    with open(FILE_STEP5, 'w', encoding='utf-8') as f:
        json.dump(step5_data, f, indent=2, ensure_ascii=False)

    # 10. Step 6: Final Logic QA
    print("\n--- Running Step 6: Final Logic QA ---")
    tasks = [run_logic_based_qa_task(s) for s in step5_data]
    step6_data = [
        r for r in await tqdm_asyncio.gather(*tasks) if r is not None
    ]

    with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(step6_data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Pipeline complete. Final output saved to {FINAL_OUTPUT}")
    print(f"Total samples processed successfully: {len(step6_data)}")


if __name__ == "__main__":
    asyncio.run(main())
