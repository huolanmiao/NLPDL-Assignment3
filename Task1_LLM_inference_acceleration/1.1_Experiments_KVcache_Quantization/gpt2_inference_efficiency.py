from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
from transformers import BitsAndBytesConfig
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


mode = 'naive'
# mode = 'KV-cache'
# mode = 'fp16'
# mode = 'int8'
# mode = 'fp4'
# mode = 'fp4-bf16'
# mode = 'fp4-double-quant'


if mode == 'naive':
    print(f"Naive mode: no KV Cache & no quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2) 
    print(f"Model Memory Usage: {model_memory} MB")
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (naïve approach, no KV-cache)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=False)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[-1]  # Exclude input tokens
    throughput_naive = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_naive = torch.cuda.max_memory_allocated()
    # print(model.get_memory_footprint() / (1024 ** 2))
    gpu_memory_naive = (memory_after_naive - model_memory) / (1024 ** 2)  
    print(f"Inference Time (Naïve): {inference_time:.4f} seconds")
    print(f"Throughput (Naïve): {throughput_naive:.4f} tokens/second")
    print(f"Inference time GPU Memory Usage (Naïve): {gpu_memory_naive:.2f} MB")
    
if mode == 'KV-cache':
    print(f"KV-cache mode: with KV Cache & no quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_kv_cache = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_kv_cache = torch.cuda.max_memory_allocated()
    gpu_memory_kv_cache = (memory_after_kv_cache - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (KV-cache): {inference_time:.4f} seconds")
    print(f"Throughput (KV-cache): {throughput_kv_cache:.4f} tokens/second")
    print(f"GPU Memory Usage (KV-cache): {gpu_memory_kv_cache:.2f} MB")
    
if mode =='fp16':
    print(f"FP16 mode: with KV Cache & with FP16 quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    # Enable FP16 for quantization
    model = GPT2LMHeadModel.from_pretrained(model_name).half().to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_fp16 = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_fp16 = torch.cuda.memory_allocated()
    gpu_memory_fp16 = (memory_after_fp16 - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (FP16): {inference_time:.4f} seconds")
    print(f"Throughput (FP16): {throughput_fp16:.4f} tokens/second")
    print(f"GPU Memory Usage (FP16): {gpu_memory_fp16:.2f} MB")

if mode == "int8":
    print(f"INT8 mode: with KV Cache & with INT8 quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    # Enable INT8 for quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=quantization_config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    # print(model.device)
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_int8 = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_int8 = torch.cuda.memory_allocated()
    gpu_memory_int8 = (memory_after_int8 - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (INT8): {inference_time:.4f} seconds")
    print(f"Throughput (INT8): {throughput_int8:.4f} tokens/second")
    print(f"GPU Memory Usage (INT8): {gpu_memory_int8:.2f} MB")

if mode == "fp4":
    print(f"FP4 mode: with KV Cache & with FP4 quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    # Enable FP4 for quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=quantization_config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    # print(model.device)
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_fp4 = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_fp4 = torch.cuda.memory_allocated()
    gpu_memory_fp4 = (memory_after_fp4 - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (FP4): {inference_time:.4f} seconds")
    print(f"Throughput (FP4): {throughput_fp4:.4f} tokens/second")
    print(f"GPU Memory Usage (FP4): {gpu_memory_fp4:.2f} MB")

if mode == 'fp4-bf16':
    print(f"FP4-BF16 mode: with KV Cache & with FP4-BF16 quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    # Enable FP4-BF16 for quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=quantization_config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    # print(model.device)
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_fp4_bf16 = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_fp4_bf16 = torch.cuda.memory_allocated()
    gpu_memory_fp4_bf16 = (memory_after_fp4_bf16 - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (FP4-BF16): {inference_time:.4f} seconds")
    print(f"Throughput (FP4-BF16): {throughput_fp4_bf16:.4f} tokens/second")
    print(f"GPU Memory Usage (FP4-BF16): {gpu_memory_fp4_bf16:.2f} MB")
    
if mode == 'fp4-double-quant':
    print(f"FP4-Double-Quant mode: with KV Cache & with FP4 quantization.")
    torch.cuda.empty_cache()  # Clear any unused memory
    memory_before = torch.cuda.memory_allocated()
    # Load model and tokenizer
    model_name = 'gpt2'
    # Enable FP4 for quantization
    double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,)
    model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=double_quant_config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Measure GPU memory before inference
    model_memory = (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
    print(f"Model Memory Usage: {model_memory} MB")
    # print(model.device)
    
    # Prepare input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    # Run inference (KV-cache approach)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'], min_length=1000, max_length=1000, use_cache=True)
    end_time = time.time()
    
    # Outputs is a sequence of token ids. We need to decode to see the text using tokenizer.
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
    # Calculate throughput
    inference_time = end_time - start_time
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[1]  # Exclude input tokens
    throughput_fp4_double_quant = tokens_generated / inference_time  # tokens per second
    # GPU memory usage
    memory_after_fp4_double_quant = torch.cuda.memory_allocated()
    gpu_memory_fp4_double_quant = (memory_after_fp4_double_quant - memory_before) / (1024 ** 2)  # GB
    print(f"Inference Time (FP4-Double-Quant): {inference_time:.4f} seconds")
    print(f"Throughput (FP4-Double-Quant): {throughput_fp4_double_quant:.4f} tokens/second")
    print(f"GPU Memory Usage (FP4-Double-Quant): {gpu_memory_fp4_double_quant:.2f} MB")
    
