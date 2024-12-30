import os
from opencompass.models import OpenAISDK
 
 
internlm_url = 'https://api.deepseek.com' # 你前面获得的 api 服务地址
# internlm_api_key = "sk-6d7cfdf9706f4629a8aead5921d82047"
internlm_api_key = "sk-e39450eb1e1d4a8e825df0a7e4f5f411"
models = [
    dict(
        # abbr='internlm2.5-latest',
        type=OpenAISDK,
        path='deepseek-chat', # 请求服务时的 model name
        # 换成自己申请的APIkey
        key=internlm_api_key, # API key
        openai_api_base=internlm_url, # 服务地址
        rpm_verbose=True, # 是否打印请求速率
        query_per_second=0.16, # 服务请求速率
        max_out_len=1024, # 最大输出长度
        max_seq_len=4096, # 最大输入长度
        temperature=0.0, # 生成温度
        batch_size=1, # 批处理大小
        retry=3, # 重试次数
    )
]