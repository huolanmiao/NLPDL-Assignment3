 # 如何做Quantization？
参考https://github.com/liuzard/transformers_zh_docs/blob/master/docs_zh/main_classes/quantization.md \
```
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=quantization_config)
```

# 如何查看内存占用情况？
```
torch.cuda.memory_summary()可以得到详细表格
torch.cuda.max_memory_allocated()可以得到到目前为止最大内存占用
model.get_memory_footprint()可以得到模型内存占用情况
```