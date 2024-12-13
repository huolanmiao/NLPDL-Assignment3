# 测试的结果保留在哪里？
```
./opencompass/outputs/
```
# 如何更改prompt？

dataset的配置文件在configs/datasets中，通过gsm8k_infer_cfg中的prompt_template设置prompt格式。\
主要参考./opencompass/configs/datasets/gsm8k/文件夹中的datasets配置文件

# 如何使用api配置model？

参考配置文件configs/models/deepseek/ \
我的models配置文件在./opencompass/configs/openai/deepseek_api.py,使用OpenAISDK.\

```python
# deekseek api
base_url = "https://api.deepseek.com"
api_key = "sk-e39450eb1e1d4a8e825df0a7e4f5f411"
model = "deepseek-chat"
```


# 如何运行测试？
Open Campass 在配置测试任务时，允许接受一个.py配置文件作为任务相关参数，需要包含models和datasets字段。\
python run.py configs/eval_demo.py \
也可以分别指定datasets和models的配置文件 \

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
# 参考
https://platform.deepseek.com/sign_in 

https://opencompass.readthedocs.io/en/latest/get_started/installation.html

