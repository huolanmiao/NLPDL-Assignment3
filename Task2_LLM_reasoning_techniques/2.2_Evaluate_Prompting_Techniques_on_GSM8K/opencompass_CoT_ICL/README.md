# 运行方法:
```
cd ./opencompass
```
## 取64条测试
```python
python run.py --models deepseek_api.py --datasets demo_gsm8k_chat_gen.py --debug
```
## few-shot COT
```python
python run.py --models deepseek_api.py --datasets gsm8k_gen_1d7fe4.py --debug 
```
## COT best prompt: "Let's think step by step."
```python
python run.py --models deepseek_api.py --datasets gsm8k_gen_701491.py --debug 
```
## COT official prompt："As an expert problem solver solve step by step the following mathematical questions."
```python
python run.py --models deepseek_api.py --datasets gsm8k_gen_cot_official.py --debug 
```
## zero-shot
```python
python run.py --models deepseek_api.py --datasets gsm8k_gen_zero_shot.py --debug 
```
# 测试结果:
- few-shot COT: **90.07**
- COT best prompt: **78.77**
- COT official prompt: **76.88**
- zero-shot: **79.23**
- log dir: **./opencompass/outputs/**

