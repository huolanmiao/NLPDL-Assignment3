# 运行方法：
```shell
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```

# 实现细节：
- KV Cache保存((key,value),...)，12层的decoder的KV，tuple length为12
- 先输入prompt，整体计算QKV，然后取最后一行hidden_state，预测下一token
- 之后每一个iteration，输入新token，在每一个layer中，计算该token的QKV，KV与KV Cache拼接，用Q和拼接后的KV得到新token在该层的hidden_state
- 模型返回下一个token的logits，和新的KV Cache。
- attention_mask在当前代码的目的仅仅是mask掉padding tokens，每个iteration因为新加一个token所以attention_mask拼接一个1
- 实际attention_mask只需要在prefill的时候，只取最后一个token的hidden_state(last row of hidden_states)即可。

# Results：

batchsize变大的时候能够节省更大的计算量，KV Cache速度有明显的提升。
```python
# bsz=1
# Time taken for golden greedy decoding without KV cache:  36.51356053352356
# Time taken for customized greedy decoding:  34.10257434844971
# bsz=10
# Time taken for golden greedy decoding without KV cache:  12.239511013031006
# Time taken for customized greedy decoding:  3.8154773712158203
# bsz=20
# Time taken for golden greedy decoding without KV cache:  11.563897371292114
# Time taken for customized greedy decoding:  2.005453109741211
```