import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from customized_gpt2 import CustomizedGPT2LMHeadModel

@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    # print(tokenized_batch['attention_mask'].shape)
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model(**tokenized_batch, use_cache=True)
        # print(outputs['logits'].shape)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        # print(output_tokens.shape) 每次生成[batchsize, 1]个新token
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)
        tokenized_batch['past_key_values'] = outputs['KV_Cache']
        # print(len(outputs['KV_Cache'])) 应为decoder层数12
        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time


@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    # print(tokenized_batch['attention_mask']) # 只mask用于padding的token
    # print(tokenized_batch['input_ids'])
    
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        # 只与hidden_states最后一行有关，也就是说predict next token取决于当前最后一个token运算来的hidden_state，所以不需要存前面token的最终的hidden_state
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True) 
        # print(tokenized_batch['attention_mask'].shape)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 1
    times = [0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")
    # print(tokenizer(".")) # ","的ids是11
    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time

        assert torch.equal(golden_wo_cache_res, custom_res), "Decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])

# export HF_ENDPOINT=https://hf-mirror.com

# bsz=1
# Time taken for golden greedy decoding without KV cache:  36.51356053352356
# Time taken for customized greedy decoding:  34.10257434844971
# bsz=10
# Time taken for golden greedy decoding without KV cache:  12.239511013031006
# Time taken for customized greedy decoding:  3.8154773712158203
# bsz=20
# Time taken for golden greedy decoding without KV cache:  11.563897371292114
# Time taken for customized greedy decoding:  2.005453109741211