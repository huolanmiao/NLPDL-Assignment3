# NLPDL - Assignment 3

## Environment 
This assignment requires only two packages:
- `torch`
- `time`
- `transformers`
- `langchain`
- `opencompass`
- `datasets`
- `re`

## 文件结构
- README.md
- Task1_LLM_inference_acceleration
  - 1.1_Experiments_KVcache_Quantization
    - gpt2_inference_efficiency.py
    - note.md
    - result.md
  - 1.2_KV_Cache_Implementation
    - LICENSE
    - README.md
    - __pycache__
    - customized_gpt2.py
    - data.txt
    - documentation.md
    - main.py
    - outputs
- Task2_LLM_reasoning_techniques
  - 2.1_RAG
    - DIY_RAG
    - LightRAG
    - RAG_note.md
    - try_api.py
  - 2.2_Evaluate_Prompting_Techniques_on_GSM8K
    - README.md
    - Reflexion
    - opencompass_CoT_ICL

## 运行方法
### Task1_LLM_inference_acceleration
```python
# Inference efficiency
python ./Task1_LLM_inference_acceleration/1.1_Experiments_KVcache_Quantization/gpt2_inference_efficiency.py

# KV Cache
cd ./Task1_LLM_inference_acceleration/1.2_KV_Cache_Implementation/
python main.py
```
### Task2_LLM_reasoning_techniques
```python
# RAG
cd ./Task2_LLM_reasoning_techniques/2.1_RAG/DIY_RAG/
python rag.py
```
```python
# Evaluation with opencompass
cd ./Task2_LLM_reasoning_techniques/2.2_Evaluate_Prompting_Techniques_on_GSM8K/opencompass_CoT_ICL/opencompass/

# 取64条测试
python run.py --models deepseek_api.py --datasets demo_gsm8k_chat_gen.py --debug

# few-shot COT
python run.py --models deepseek_api.py --datasets gsm8k_gen_1d7fe4.py --debug 

# few-shot-4 90.07
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot.py --debug 
# few-shot-4 shuffle 89.46
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_shuffle.py --debug 
# few-shot-8 89.16
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_8.py --debug 
# few-shot-8 shuffle 89.01
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_8_shuffle.py --debug 
# few-shot-8 generated default 90.30
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_generated_default.py --debug 
# few-shot-8 generated epoch10 lr1e-2  90.37
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_generated_epoch10_lr_1e-2.py --debug 
# few-shot-8 randomly sample from example pool 88.93
python run.py --models deepseek_api.py --datasets gsm8k_gen_few_shot_8_random_sample.py --debug 

# COT best prompt: "Let's think step by step."
python run.py --models deepseek_api.py --datasets gsm8k_gen_701491.py --debug 

# COT official prompt："As an expert problem solver solve step by step the following mathematical questions."
python run.py --models deepseek_api.py --datasets gsm8k_gen_cot_official.py --debug 

# zero-shot
python run.py --models deepseek_api.py --datasets gsm8k_gen_zero_shot.py --debug 
```
```python
# Evaluation with opencompass
cd ./Task2_LLM_reasoning_techniques/2.2_Evaluate_Prompting_Techniques_on_GSM8K/Reflexion/
python reflexion.py
```