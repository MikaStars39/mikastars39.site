数据作为所有模型公司的核心技术，目前已知的千亿级别模型的数据大盘基本属于未知状态。这里根据参考公开report和神秘来源大概对模型sft-rl全流程的数据recipe进行以下的大致预估。主要几个要点：
- SFT, No thinking模式和thinking模式应该使用一套系统进行数据评估和整理
- 总体干净和严格验证的数据规模预估需要达到1M以上
- 不是越多越好，在cover一定case的情况下应该做更多的筛选和分级，选出更高质量的数据
- 做难度分级，e.g.，raw - dpsk 32b distill - r1 三级别难度分级filtering

# 数据分类与验证方案

### 1. Code (代码)
* **Code Generation, File Level (OJ)**
    - **具体细节/验证方式：** 多语言 sandbox
    - **prompt / answer 来源：** 开源 + 可能需要购买
* **Code Generation, Repo Level, Web Dev (SWE)**
    - **具体细节/验证方式：** checklist i.e., vlm, gemRM + sandbox
    - **prompt / answer 来源：** 开源 + 可能需要购买
* **Code Edit, File Level (SWE)**
    - **具体细节/验证方式：** sandbox
    - **prompt / answer 来源：** 开源 + 可能需要购买
* **Code Edit, Repo Level (SWE)**
    - **具体细节/验证方式：** sandbox
    - **prompt / answer 来源：** 开源 + 可能需要购买

---

### 2. STEM
* **Math (数学)**
    - **具体细节/验证方式：** rule-based verifier, cot filter, genRM
    - **prompt / answer 来源：** 基本可以开源获取，开源数据总量对于训练合适
* **General STEM**
    - **具体细节/验证方式：** answer verifier
    - **prompt / answer 来源：** 开源 + 购买

---

### 3. Logic (逻辑)
* **具体细节/验证方式：** answer verifier, python based check
* **prompt / answer 来源：** 收集 + 合成

---

### 4. IF (指令遵循)
* **具体细节/验证方式：** answer verifier, python based check e.g., regex
* **prompt / answer 来源：** 收集 + 合成

---

### 5. General Chat (通用对话)
* **中英语**
    - **具体细节/验证方式：** genRM
    - **prompt / answer 来源：** 开源 + 收集
* **小语种**
    - **具体细节/验证方式：** genRM
    - **prompt / answer 来源：** 开源 + 收集

---

### 6. Alignment (对齐)
* **具体细节/验证方式：** genRM, xxxguard model
* **prompt / answer 来源：** 开源 + 收集

---

### 7. Agentic (智能体)
* **具体细节/验证方式：** genRM
* **prompt / answer 来源：** 开源 + 收集 + 合成

## STILL
- [RUC-AIBOX/STILL-3-Preview-RL-Data](https://huggingface.co/datasets/RUC-AIBOX/STILL-3-Preview-RL-Data)
- [RUC-AIBOX/STILL-3-RL-90K](https://huggingface.co/datasets/RUC-AIBOX/STILL-3-RL-90K)
- [RUC-AIBOX/STILL-3-TOOL-32B-Data](https://huggingface.co/datasets/RUC-AIBOX/STILL-3-TOOL-32B-Data)

## OpenThoughts
- [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [open-thoughts/OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)
- [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- [open-thoughts/OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- [open-thoughts/OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL)

## PrimeIntellect
- [PrimeIntellect/SYNTHETIC-2](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2)
- [PrimeIntellect/INTELLECT-2-only-math-filtered-2k](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-2-only-math-filtered-2k)
- [PrimeIntellect/Multi-SWE-RL](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL)
- [PrimeIntellect/INTELLECT-3-SFT](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-SFT)
- [PrimeIntellect/INTELLECT-3-RL](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL)

## Skywork
- [Skywork/Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)
- 
## Olmo
- [allenai/Dolci-Think-RL-32B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B)
- [allenai/Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B)
- [allenai/Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)
- [allenai/Dolci-RL-Zero-General-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-General-7B)
- [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use)

## Nvidia / Nemotron
- [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)
- [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)
- [nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1)
- [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following)
- [nvidia/Nemotron-RL-knowledge-openqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-openqa)
- [nvidia/Nemotron-RL-knowledge-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa)
- [nvidia/Nemotron-RL-coding-competitive_coding](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding)
- [nvidia/Nemotron-Cascade-SFT-SWE](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-SWE)
- [nvidia/AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math)

## Nex-agi
- [nex-agi/agent-sft](https://huggingface.co/datasets/nex-agi/agent-sft)

## Agentica
- [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- [agentica-org/DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)

## Huggingface
- [HuggingFaceH4/Polaris-Dataset-53K](https://huggingface.co/datasets/HuggingFaceH4/Polaris-Dataset-53K)
- [HuggingFaceH4/OpenR1-Math-220k-default-verified](https://huggingface.co/datasets/HuggingFaceH4/OpenR1-Math-220k-default-verified)

## PRIME-RL
- [PRIME-RL/Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data)
- [PRIME-RL/Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data)

## DAPO-Math
- [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

## Kwai
- [Kwai-Klear/KlearReasoner-MathSub-30K](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-MathSub-30K)

## Open-Reasoner-Zero
- [Open-Reasoner-Zero/orz_math_72k_collection_extended](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_72k_collection_extended)

## Dr. Tulu
- [rl-research/dr-tulu-sft-data](https://huggingface.co/datasets/rl-research/dr-tulu-sft-data)
- [rl-research/dr-tulu-rl-data](https://huggingface.co/datasets/rl-research/dr-tulu-rl-data)

## hotpotqa
- [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

## VerIF
- [https://github.com/THU-KEG/VerIF](https://github.com/THU-KEG/VerIF)

## SCU

[virtuoussy/Multi-subject-RLVR](https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR)
