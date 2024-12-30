# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-e39450eb1e1d4a8e825df0a7e4f5f411", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    max_tokens=10,
    temperature=0.7,
    stream=False,
    logprobs = True,
)

# ans_index = rsp[i].choices[0].logprobs["tokens"].index(" "+ sttr.group(0)[14:-1])
# prob = np.exp(rsp[i]['choices'][0]["logprobs"]["token_logprobs"][ans_index])
# loss = -np.log(prob) * FACTOR
# all_losses.append(loss)
print(response.choices[0].logprobs.tokens)
# print(response.choices[0].message.content)